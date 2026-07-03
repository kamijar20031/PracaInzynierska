import tensorflow as tf
import numpy as np
import pickle
import os

height = 32
width = 128

# wczytanie slownika
vocab = pickle.load(open('vocab.pkl', 'rb'))
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None, oov_token="[UNK]")

# odwrocenie tej samej funkcji
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, oov_token="[UNK]", invert=True
)

# ładowanie zapisanego modelu
model = tf.keras.models.load_model("model.keras", compile=False)


def loadImage(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, height, width)
    image = tf.cast(image, tf.float32) / 255.0
    # expand dims jest wazne bo juz nie bedzie paddingu
    image = tf.expand_dims(image, axis=0)
    return image,

# 5. Funkcja dekodująca wyjście z sieci (CTC Decode)
def decodeBatchPredictions(pred):
    input_shape = tf.shape(pred)
    input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(input_shape[1], "int32")
    results = tf.keras.backend.ctc_decode(pred, input_length=input_length, greedy=True)[0][0]
    output_text = []
    for res in results:
        res = res[res != -1]
        chars = num_to_char(res)
        text = tf.strings.reduce_join(chars).numpy().decode("utf-8")
        output_text.append(text)
    return output_text

if __name__ == "__main__":
   
    imgs = os.listdir("slowa")
    for img in imgs:
        try:
            imgL = loadImage("slowa/" + img)
            preds = model.predict(imgL)
            decoded_text = decodeBatchPredictions(preds)
            print(f"\nRozpoznany tekst dla {img}:")
            print(f"-> {decoded_text[0]}")
            
        except Exception as e:
            print(f"Wystąpił błąd: {e}")