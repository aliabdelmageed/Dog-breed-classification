from flask import Flask, request, jsonify, render_template, request
from flask_cors import CORS, cross_origin
from keras.optimizers import Adam
from keras.applications import imagenet_utils
import cv2
import numpy as np
from keras.layers import Activation, Dropout, Input
from keras.models import Sequential, Model, load_model
from keras.applications import Xception
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
import logging
import json
import urllib.request
from urllib.request import Request, urlopen
import base64
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

global model
app = Flask(__name__,)
CORS(app)
def load_img(url):
    image_url = url.split(',')[1]
    image_url = image_url.replace(" ", "+")
    image_array = base64.b64decode(image_url)
    image_array = np.frombuffer(image_array, np.uint8)
    image_array = cv2.imdecode(image_array, -1)
    return image_array  

def build_finetune_model():
    inputs = Input((300, 300, 3))
    backbone = Xception(input_tensor=inputs, include_top=False, weights='imagenet')
    backbone.trainable = False
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(120, activation="softmax")(x)
    model = Model(inputs, x)
    model.load_weights('weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    print("Model is loaded")                                    
    return model


model = build_finetune_model()

def predict(img):
    img = cv2.resize(img,(300,300))
    img = np.reshape(img,[1,300,300,3])
    img= preprocess_input(img)
    preds = model.predict(img)
    return preds.argmax()

'''
def predict_mob(img):
    img = cv2.resize(img,(300,300))
    img = np.reshape(img,[1,300,300,3])
    #classes = model.predict(img)[0]
    #return class_list[classes.argmax()],classes
    return str(classes.argmax())
'''

@app.route('/classify', methods=['GET'])
def classify():
    image_url = request.args.get('imageurl')
    image_array = load_img(image_url)
    class_index = predict(image_array)
    my_file = open("classes.txt", "r")
    content = my_file.read()
    classes = content.rsplit()
    my_file.close()
    class_name = classes[int(class_index)]
    print(class_name)
    result = []
    result.append({"class_name":class_name})
    return jsonify({'results':result})


@app.route('/', methods=['GET','POST'])
def test():
    if  request.method == 'POST' and 'GET':
        nparr = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        with graph.as_default():
            prediction = predict_mob(img)
        ret = prediction
        render_template('index.html')
        return ret
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host = '127.0.0.6',debug=False)                   