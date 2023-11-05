from flask import Flask,render_template,request

import numpy as np
#importing ImageDataGenerator to convert image into numerical data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# importing Sequential CNN model
from tensorflow.keras.models import Sequential
import cv2
# importing Conv2D for creting convolution layer and MaxPooling2D for pooling
from tensorflow.keras.layers import Conv2D,MaxPooling2D
# importing Flatten to convert 2D Array into 1D to create Dense Layer, Dropout is used to overcome overfitting.
from tensorflow.keras.layers import Flatten,Dense,Activation,Dropout
# importing Adam optimizer
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import BatchNormalization


from keras.preprocessing import image


import os

from keras.models import model_from_json

# load json and create model
json_file = open('mainModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
mainModel = model_from_json(loaded_model_json)

# load weights into new model
mainModel.load_weights("mainModel.h5")
print("Loaded model from disk")

app = Flask(__name__)

import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))  # Bind to any available port on the loopback interface
        return s.getsockname()[1]

# Example of how to use the function
port = find_free_port()


@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(224, 224), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = mainModel.predict(img_array)
    predicted_class = np.argmax(predictions)
    emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    predicted_emotion = emotion_labels[predicted_class]
    
    return render_template('index.html',prediction = predicted_emotion,filename = "/Users/reddy/Desktop/RK20UG45_Kunduru Jayasimha Reddy/images/"+imagefile.filename)


if __name__ == '__main__':
    app.run(debug = False,port = port)


