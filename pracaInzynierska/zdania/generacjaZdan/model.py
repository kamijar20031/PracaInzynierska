import tensorflow as tf
import pickle

# Włączamy memory growth dla kart, czyli nie rezerwujemy od razu całego VRAMu karty, tylko używamy tyle ile potrzebujemy. Na moim serwerze to niczego nie da, ale np gdybym to chciał to uruchomić na czymś z GUI albo systemie ktory juz jakoś wykorzystuje karte graficzną to miałbym problem bez czegoś takiego
try: 
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: 
    pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import layers
from keras.models import Model
import os
import typing
import importlib
import numpy as np
import logging
from abc import ABC
from abc import abstractmethod
from PIL import Image as PilImage
import time
import queue
import threading
from enum import Enum
from keras.callbacks import Callback
from keras.metrics import Metric
import tarfile
from io import BytesIO
from zipfile import ZipFile

PADDING_TOKEN = -1
DATASET_NAME = "temp"
height = 32
width = 512
batch_size = 16
learning_rate = 0.0005
train_epochs = 500
train_workers = 20
split = 0.9
dataset, vocab, maxLength = [], set(), 0
options = tf.data.Options()
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True


# CER i WER zabrane z MLTU. Metryka która liczy to ile znakow trzeba podmienic żeby predykcja była poprawna (C) i to czy w ogóle słowa są błędne (W)


class CERMetric(tf.keras.metrics.Metric):
    def __init__(self, padding_token, name="CER", **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0.0, name="batch_counter", dtype=tf.float32)
        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.shape(y_pred)
        batch_size = input_shape[0]
        input_length = tf.fill([batch_size], input_shape[1])
        decode_predicted, _ = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)
        decoded = decode_predicted[0]
        decoded_length = tf.math.count_nonzero(tf.not_equal(decoded, -1), axis=1, dtype=tf.int32)
        true_length = tf.math.count_nonzero(tf.not_equal(y_true, self.padding_token), axis=1, dtype=tf.int32)
        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decoded, decoded_length)
        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, true_length), tf.int64)
        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))
        true_labels_sparse = tf.sparse.retain(true_labels_sparse, tf.not_equal(true_labels_sparse.values, self.padding_token))
        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.batch_counter.assign_add(tf.cast(batch_size, tf.float32))

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.batch_counter.assign(0.0)

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.batch_counter)

import tensorflow as tf

class WERMetric(tf.keras.metrics.Metric):
    def __init__(self, padding_token, name="WER", **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0.0, name="batch_counter", dtype=tf.float32)
        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.shape(y_pred)
        batch_size = input_shape[0]
        input_length = tf.fill([batch_size], input_shape[1])
        decode_predicted, _ = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)
        decoded = decode_predicted[0]
        decoded_length = tf.math.count_nonzero(tf.not_equal(decoded, -1), axis=1, dtype=tf.int32)
        true_length = tf.math.count_nonzero(tf.not_equal(y_true, self.padding_token), axis=1, dtype=tf.int32)
        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decoded, decoded_length)
        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, true_length), tf.int64)
        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))
        true_labels_sparse = tf.sparse.retain(true_labels_sparse, tf.not_equal(true_labels_sparse.values, self.padding_token))
        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)
        word_errors = tf.cast(tf.not_equal(distance, 0), tf.float32)
        self.wer_accumulator.assign_add(tf.reduce_sum(word_errors))
        self.batch_counter.assign_add(tf.cast(batch_size, tf.float32))

    def reset_state(self):
        self.wer_accumulator.assign(0.0)
        self.batch_counter.assign(0.0)

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.batch_counter)

# Obsługa CTCLoss
class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, padding_token):
        super().__init__()
        self.padding_token = padding_token
    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        time_steps = tf.shape(y_pred)[1]
        input_length = tf.fill([batch_size, 1], tf.cast(time_steps, tf.int64))
        label_length = tf.math.count_nonzero(tf.not_equal(y_true, self.padding_token), axis=1, keepdims=True)
        label_length = tf.cast(label_length, tf.int64)
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def loadImage(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, height, width)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def encodeLabel(image, label):
    chars = tf.strings.unicode_split(label,"UTF-8")
    label = char_to_num(chars)
    return image, label

def convBlock(prev, filter, strides):
    x = layers.Conv2D(filter, 3, padding = "same", strides = strides, kernel_initializer="he_uniform")(prev)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def randomBrightness(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

rotation_layer = tf.keras.layers.RandomRotation(0.05)
def randomRotate(image, label):
    image = rotation_layer(image, training=True)
    return image, label

def randomErodeDilate(image, label):
    image = tf.expand_dims(image, 0)
    image = tf.cond(
        tf.random.uniform([]) < 0.5,
        lambda: tf.nn.max_pool2d(image, 2, 1, "SAME"),
        lambda: tf.nn.avg_pool2d(image, 2, 1, "SAME"),
    )
    return tf.squeeze(image, 0), label

def randomSharpen(image, label):
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

def augment(image, label):
    image, label = randomRotate(image, label)
    # image, label = randomErodeDilate(image, label)
    image, label = randomSharpen(image, label)
    image, label = randomBrightness(image, label)
    return image, label

# Procesowanie datasetu wedlug struktury IAM_Words, potem bedzie zastapione moim jak uda mi sie go odpowiednio dopracowac
words = open(os.path.join(DATASET_NAME,"words.txt"), "r").readlines()
for line in words:
    splitS = line.split(" ----- ")
    file = splitS[1].rstrip('\n')
    label = splitS[0].rstrip('\n')
    dataset.append([file, label])
    vocab.update(list(label))
    if len(label)>maxLength:
        maxLength = len(label)


paths = np.array([x[0] for x in dataset], dtype=str)
labels = np.array([x[1] for x in dataset], dtype=str)
idx = np.arange(len(paths))
np.random.shuffle(idx)
paths = paths[idx]
labels = labels[idx]
split_idx = int(len(paths) * split)
train, val = paths[:split_idx], paths[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

train_dataset = tf.data.Dataset.from_tensor_slices((train, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val, val_labels))

train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

vocab = sorted(vocab)
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None, oov_token="[UNK]")

num_classes = len(char_to_num.get_vocabulary()) + 1 
pickle.dump(char_to_num.get_vocabulary(), open('vocab.pkl', 'wb'))

# Augmentacja danych przygotowanymi funkcjami
train_dataset = (train_dataset.map(loadImage, num_parallel_calls=tf.data.AUTOTUNE).map(encodeLabel, num_parallel_calls=tf.data.AUTOTUNE).map(augment, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(
    batch_size,
    padded_shapes=(
        (height, width, 3),
        (None,)
    ),
    padding_values=(
        tf.constant(0.0, dtype=tf.float32),     # obraz
        tf.constant(PADDING_TOKEN, dtype=tf.int64) # label
    )
).prefetch(tf.data.AUTOTUNE))
val_dataset = (val_dataset.map(loadImage, num_parallel_calls=tf.data.AUTOTUNE).map(encodeLabel, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(
    batch_size,
    padded_shapes=(
        (height, width, 3),
        (None,)
    ),
    padding_values=(
        tf.constant(0.0, dtype=tf.float32),     # obraz
        tf.constant(PADDING_TOKEN, dtype=tf.int64) # label
    )
).prefetch(tf.data.AUTOTUNE))




# Definicja modelu
inputs = layers.Input(shape=(height, width, 3), name="input")
x = convBlock(inputs, 32, 1, True)
x = layers.MaxPooling2D((2,2))(x) 
x = convBlock(x, 64, 1, True)
x = convBlock(x, 128, 1, False)
x = convBlock(x, 256, 1, True)
x = convBlock(x, 512, 1, False)
squeezed = layers.Permute((2, 1, 3))(x)
squeezed = layers.Reshape((x.shape[2], x.shape[1]*x.shape[3]))(squeezed)
blstm = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(squeezed)
blstm = layers.Dropout(0.3)(blstm)
blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(blstm)
blstm = layers.Dropout(0.3)(blstm)
output = layers.Dense(num_classes, activation="softmax", name="output")(blstm)

# Kompilacja modelu
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=CTCLoss(padding_token=PADDING_TOKEN), metrics=[CERMetric(padding_token=PADDING_TOKEN), WERMetric(padding_token=PADDING_TOKEN)])

earlystopper = EarlyStopping(monitor="val_CER", patience=20, mode="min",verbose=1)
tb_callback = TensorBoard("logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="min")
checkpoint = tf.keras.callbacks.ModelCheckpoint("model.keras", monitor="val_CER", mode="min", save_best_only=True, verbose=1)

model.fit(train_dataset.repeat(), verbose=1, validation_data=val_dataset.repeat(), epochs=train_epochs, steps_per_epoch=len(train)//batch_size, validation_steps=len(val)//batch_size, callbacks=[earlystopper, reduceLROnPlat, tb_callback, checkpoint])