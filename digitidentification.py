import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

print(tf.__version__)

traindatagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = traindatagen.flow_from_directory(
    "./trainingSet",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical"
)
testdatagen = ImageDataGenerator(rescale=1. / 255)
test_set = testdatagen.flow_from_directory(
    "./testSet",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical"
)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

cnn.add(tf.keras.layers.Dense(units=10, activation="softmax"))

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
cnn.fit(x=training_set, validation_data=test_set, epochs=10)

model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
cnn.save_weights("model.h5")
print("Saved model to disk")
