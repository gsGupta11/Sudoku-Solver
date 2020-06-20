import tensorflow as tf
import numpy as np
from keras.models import load_model,model_from_json
import imagesplitting
import cv2
import pickle

img = cv2.imread("./testimages/test5.png", 1)                   ###69,75,81
toidentifyarr,matrixdic,x = imagesplitting.splitFunc(img)

fourth_model = load_model('../fourth.h5')

json_file = open('../model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("../model.h5")
print("Loaded model1 from disk")


mnistjson_file = open('../mnist.json', 'r')
mnistloaded_model_json = mnistjson_file.read()
mnistjson_file.close()
loaded_model_mnist = tf.keras.models.model_from_json(mnistloaded_model_json)
loaded_model_mnist.load_weights("../mnist.h5")
print("Loaded model2 from disk")


final_json = open("../final.json","r")
final_loaded_json = final_json.read()
final_json.close()
final_model = tf.keras.models.model_from_json(final_loaded_json)
final_model.load_weights("../final.h5")
print("Loaded Model 3")

knn_model = open("../knn.model", 'rb')
model_to_use = pickle.load(knn_model)
knn_model.close()

#
# try_try_json = open("../trytry.json","r")
# try_loaded_json = try_try_json.read()
# try_try_json.close()
# try_model = tf.keras.models.model_from_json(try_loaded_json)
# try_model.load_weights("../trytry.h5")
# print("Loaded Model 4")

print(toidentifyarr)

for i in toidentifyarr:
    test_image=x[i][11:,12:]
    # img = colorcelldict[i][10:,10:]
    test_image = cv2.resize(test_image,(30,30))
    test_image = cv2.bitwise_not(test_image)
    # test_image = cv2.dilate(test_image, np.ones((2,2), np.uint8),iterations=1)
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    # test_image = cv2.GaussianBlur(test_image,(5,5),0)
    # _, test_image = cv2.threshold(test_image, 75, 255, cv2.THRESH_BINARY)
    test_image =cv2.cvtColor(test_image,cv2.COLOR_BGR2HSV)
    lb=np.array([0,0,30])
    ub=np.array([80,80,255])
    test_image = cv2.inRange(test_image,lb,ub)
    
    forfourth = test_image.copy()
    forfourth = cv2.resize(test_image, (28, 28))
    kernelforfourth = np.ones((1,1), np.uint8)
    # dilatedfouth = cv2.dilate(forfourth,kernel=kernelforfourth,iterations=4)

    # knnimg = cv2.resize(test_image,(30,30))
    # knndilated = cv2.dilate(knnimg,kernel=kernelforfourth,iterations=4)
    # knndilated = cv2.bitwise_not(knndilated)
    # resknn = str(model_to_use.predict([knndilated.flatten()])[0])
    # print(resknn)
    # forfourth = cv2.bitwise_not(dilatedfouth)
    forfourth = np.expand_dims(forfourth,axis=0)
    forfourth = np.expand_dims(forfourth, axis=-1)
    forfourth = forfourth/255.0
    # restry = try_model.predict([forfourth])[0]
    # print(restry)
    res = fourth_model.predict([forfourth])[0]
    res =np.argmax(res)
    # print(res)

    test_image = cv2.resize(test_image,(64,64))         #27,51,54,74
    blackimg = np.zeros((64,64,3),np.uint8)
    whiteimg = np.ones((64,64,3),np.uint8)*255
    test_image = cv2.bitwise_or(blackimg,whiteimg,mask=test_image)
    kernel = np.ones((1,1),np.uint8)
    # test_image = np.uint8(test_image)
    # _,test_image = cv2.threshold(test_image,200,255,cv2.THRESH_BINARY)
    # test_image = cv2.dilate(test_image, kernel=kernel, iterations=4)
    test_image1 = cv2.bitwise_not(test_image)

    cv2.imshow(i,test_image)
    test_image = np.expand_dims(test_image,axis=0)
    test_image1 = np.expand_dims(test_image1,axis=0)

    ## Typed Dataset (For 1 )
    result1 = list(loaded_model.predict(test_image1)[0])

    ## Mnist Dataset (FoR 5,6,3,2,9)
    result2 = list(loaded_model_mnist.predict(test_image)[0])

    final_result = list(final_model.predict(test_image1)[0])

    if result1.index(max(result1)) == result2.index(max(result2)) and result2.index(max(result2))==final_result.index(max(final_result)):
        print(result1.index(max(result1)))
    else:
        print(result1.index(max(result1)),result2.index(max(result2)),final_result.index(max(final_result)),res)
    cv2.waitKey(0)