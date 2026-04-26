import os, shutil, pathlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import losses
from keras import ops
from keras import optimizers
from keras.optimizers import schedules
from keras import metrics
from keras.applications.imagenet_utils import decode_predictions

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.layers import Dropout, Dense, GlobalAveragePooling1D, Flatten

from keras import layers
from keras.layers import Lambda


def get_patches(images, patch_size=16):
    input_shape = tf.shape(images)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    num_patches_h = height//patch_size
    num_patches_w = width//patch_size
    patches = tf.image.extract_patches(images=images, sizes=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], rates = [1,1,1,1], padding='VALID')
    patches= tf.reshape(tf.cast(patches, dtype=tf.float32), (batch_size, num_patches_h*num_patches_w, patch_size*patch_size*channels,))
    return patches

def preprocess_data(image, label):
    label = tf.one_hot(label, classes)
    return image, label

def plot_loss_curves(history):
    plt.clf()
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

def plot_acc_curves(history):
    plt.clf()
    history_dict = history.history
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("acc.png")

def augment_data(image):
    x = keras.layers.RandomFlip("horizontal")(image)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.2)(x)
    return x

def get_features_and_labels(dataset):
    all_feat = []
    all_lab = []
    for images, labels in dataset:
        preprocess_images = keras.applications.resnet50.preprocess_input(images)
        feats = resnet50_base.predict(preprocess_images)
        all_feat.append(feats)
        all_lab.append(labels)
    return np.concatenate(all_feat), np.concatenate(all_lab)

keras.utils.set_random_seed(42)

base = pathlib.Path("handbags-shoes")

if not os.path.exists(base / 'train' / 'handbags'):
    for category in ('handbags', 'shoes'):
        fnames = os.listdir(base/category)
        dir = base / 'train' / category
        os.makedirs(dir)
        for fname in fnames[:50]:
            shutil.copyfile(src=base/category/fname, dst = dir/fname)
        dir = base/'validation'/category
        os.makedirs(dir)
        for fname in fnames[50:75]:
            shutil.copyfile(src=base/category/fname, dst=dir/fname)
        dir = base/'test'/category
        os.makedirs(dir)
        for fname in fnames[75:]:
            shutil.copyfile(src=base/category/fname, dst=dir/fname)

train_dataset = keras.utils.image_dataset_from_directory(base/'train', image_size=(224,224), batch_size=32)
validation_dataset = keras.utils.image_dataset_from_directory(base/'validation', image_size=(224,224), batch_size=32)
test_dataset = keras.utils.image_dataset_from_directory(base/'test', image_size=(224,224), batch_size=32)

classes = 2

# train_dataset = train_dataset.map(preprocess_data)
# validation_dataset = validation_dataset.map(preprocess_data)
# test_dataset = test_dataset.map(preprocess_data)

# patch_size = 16
# image_size = 224


# for images, _ in train_dataset.take(1):
#     resized_image = keras.ops.image.resize(keras.ops.convert_to_tensor([images[0]]), size=(image_size, image_size))
#     no_channels = keras.ops.shape(resized_image)[-1]

# patches = get_patches(resized_image)

# num_patches = (image_size//patch_size) ** 2
# projection_dim = 64
# num_heads = 8
# transformed_units = [
#     projection_dim*2,
#     projection_dim
# ]
# num_transformer_layers = 2
# mlp_head_units = [
#     512,
#     256,
# ]

# input_shape = (image_size, image_size, 3)

# inputs = keras.Input(shape=input_shape)

# data_augmentation = keras.Sequential([
#     keras.layers.RandomFlip("horizontal"),
#     keras.layers.RandomRotation(0.1),
#     keras.layers.RandomZoom(0.1),
#     keras.layers.RandomTranslation(0.1,0.1),
# ])
# x = data_augmentation(inputs)
# patches = Lambda(get_patches, output_shape=(num_patches, patch_size*patch_size*3))(x)

# projection = layers.Dense(units=projection_dim)
# position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
# encoded_patches = projection(patches) + position_embedding(keras.ops.expand_dims(keras.ops.arange(start=0, stop=num_patches, step=1), axis=0))

# for _ in range(num_transformer_layers):
#     x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#     attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout = 0.1)(x1,x1)
#     x2 = layers.Add()([attention_output, encoded_patches])
#     x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
#     for units in transformed_units:
#         x3 = layers.Dense(units, activation=keras.activations.gelu)(x3)
#         x3 = layers.Dropout(0.5)(x3)
#     encoded_patches = layers.Add()([x3,x2])

# representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
# representation = layers.Flatten()(representation)
# representation= layers.Dropout(0.5)(representation)

# for units in mlp_head_units:
#     representation=layers.Dense(units, activation=keras.activations.gelu)(representation)
#     representation = layers.Dropout(0.5)(representation)

# logits = layers.Dense(classes)(representation)
# model = keras.Model(inputs=inputs, outputs=logits)

# optimizer = keras.optimizers.AdamW(learning_rate = 0.001)

# Czesc pierwsza, czyli polaczenie modelu z backbone


# model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics = [keras.metrics.CategoricalAccuracy(name="accuracy"),],)

# print(model.summary())

# sample_input = tf.zeros((32, image_size, image_size, 3))
# model(sample_input)

# history = model.fit(train_dataset, validation_data=validation_dataset, epochs=40, batch_size=32)

# Finetuning, nie dziala bo Keras ma kurwa skill issue

# backbone = keras_hub.models.Backbone.from_preset("vit_base_patch16_224_imagenet")
# preprocessor = keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
#     "vit_base_patch16_224_imagenet"
# )
# data_augmentation = keras.Sequential([
#     keras.layers.RandomFlip("horizontal"),
#     keras.layers.RandomRotation(0.1),
#     keras.layers.RandomZoom(0.1),
#     keras.layers.RandomTranslation(0.1, 0.1),
# ])

# inputs = keras.Input(shape=(224, 224, 3))  # Input shape for the images
# x = data_augmentation(inputs)              # Apply data augmentation
# x = preprocessor(x)                   # Preprocess inputs for ViT
# x = backbone(x, training=True)             # Pass through the ViT backbone
# x = Flatten()(x)                           # Flatten the ViT outputs
# x = Dense(backbone.hidden_dim, activation="linear")(x)  # Project down to the hidden dimension
# x = Dropout(0.5)(x)                        # Add dropout for regularization
# x = Dense(256, activation="relu")(x)       # Add a dense layer for feature learning
# x = Dropout(0.5)(x)                        # Another dropout layer
# outputs = Dense(2, activation="softmax")(x)  # Final classification layer

# model = keras.Model(inputs, outputs)

# model.summary()

# backbone.trainable = False

# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )

# history = model.fit(
#     train_dataset,  # Replace with your dataset
#     validation_data=validation_dataset,
#     epochs=2,  # Train for a few epochs initially
#     batch_size=32
# )


input = keras.Input(shape=(224,224,3))

h = keras.layers.Rescaling(1./255)(input)
h = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_1")(h)
h = keras.layers.MaxPool2D()(h)
h = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_2")(h)
h = keras.layers.MaxPool2D()(h)
h = keras.layers.Flatten()(h)

output = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(input,output)
print(model.summary())

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
history = model.fit(train_dataset, epochs=20, validation_data=validation_dataset)

plot_loss_curves(history)
plot_acc_curves(history)

input = keras.Input(shape=(224, 224, 3))

h = keras.layers.RandomFlip("horizontal")(input)
h = keras.layers.RandomRotation(0.1)(h)
h = keras.layers.RandomZoom(0.2)(h)
h = keras.layers.Rescaling(1./255)(h)

h = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_1")(h)
h = keras.layers.MaxPool2D()(h)
h = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_2")(h)
h = keras.layers.MaxPool2D()(h)
h = keras.layers.Flatten()(h)

output = keras.layers.Dense(1, activation="sigmoid")(h)
model = keras.Model(input, output)

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

resnet50_base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))

train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)

input = keras.Input(shape=(7,7,2048))

h = keras.layers.Flatten()(input)
h = keras.layers.Dense(256, activation="relu")(h)
h = keras.layers.Dropout(0.5)(h)

output = keras.layers.Dense(1, activation="sigmoid")(h)

model = keras.Model(input, output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
history = model.fit(train_features, train_labels, epochs=10, validation_data=(val_features, val_labels))

# Camera Capture code snippet courtesy Google Colab
# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# from base64 import b64decode

# def take_photo(filename='photo.jpg', quality=0.8):
#   js = Javascript('''
#     async function takePhoto(quality) {
#       const div = document.createElement('div');
#       const capture = document.createElement('button');
#       capture.textContent = 'Capture';
#       div.appendChild(capture);

#       const video = document.createElement('video');
#       video.style.display = 'block';
#       const stream = await navigator.mediaDevices.getUserMedia({video: true});

#       document.body.appendChild(div);
#       div.appendChild(video);
#       video.srcObject = stream;
#       await video.play();

#       // Resize the output to fit the video element.
#       google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

#       // Wait for Capture to be clicked.
#       await new Promise((resolve) => capture.onclick = resolve);

#       const canvas = document.createElement('canvas');
#       canvas.width = video.videoWidth;
#       canvas.height = video.videoHeight;
#       canvas.getContext('2d').drawImage(video, 0, 0);
#       stream.getVideoTracks()[0].stop();
#       div.remove();
#       return canvas.toDataURL('image/jpeg', quality);
#     }
#     ''')
#   display(js)
#   data = eval_js('takePhoto({})'.format(quality))
#   binary = b64decode(data.split(',')[1])
#   with open(filename, 'wb') as f:
#     f.write(binary)
#   return filename

# def predict_image(im):
#   img = keras.preprocessing.image.load_img(im, target_size=(224,224))
#   arr = keras.preprocessing.image.img_to_array(img)
#   arr = keras.applications.resnet50.preprocess_input(arr)
#   arr = np.expand_dims(arr, axis=0)
#   arr = resnet50_base(arr)
#   pred = model.predict(arr)
#   pred = "SHOE" if pred > 0.5 else "HANDBAG"
#   print("************************************\n\n")
#   print(f"...........it is a {pred}!")
#   print("\n\n************************************\n\n")

# # Camera Capture code snippet courtesy Google Colab

# from IPython.display import Image
# try:
#   filename = take_photo()
#   print('Saved to {}'.format(filename))

#   # Show the image which was just taken.
#   display(Image(filename))
#   predict_image(filename)
# except Exception as err:
#   # Errors will be thrown if the user does not have a webcam or if they do not
#   # grant the page permission to access it.
#   print(str(err))