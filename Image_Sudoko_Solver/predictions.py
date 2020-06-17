import tensorflow as tf
import numpy as np
from keras.preprocessing import image

json_file = open('../model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("../model.h5")
print("Loaded model from disk")


test_image = image.load_img("../testSet/5/img_162.jpg",target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = loaded_model.predict(test_image)
print(result)

