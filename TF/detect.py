import os
from re import X 
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from tensorflow.keras.preprocessing import image
import cv2                
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm
from glob import glob
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import json
# define ResNet50 model
def load_model(model_path):
    """Load a saves model"""
    print(f"Loading model from {model_path}")
    model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer},compile=False)
    return model

face_cascade = cv2.CascadeClassifier('static/model/haarcascades/haarcascade_frontalface_alt.xml')   
bottleneck_features = np.load('static/model/DogResnet50Data.npz')
ResNet_model=load_model("static/model/best_adamax.hdf5")
df1=pd.read_csv('static/dog_names.csv')
dog_names=df1['breed'].tolist()
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(image_url):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(image_url))
    return np.argmax(ResNet50_model.predict(img))

def path_to_tensor(image_url):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(image_url, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def dog_detector(image_url):
    prediction = ResNet50_predict_labels(image_url)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(image_url):
    img = cv2.imread(image_url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0    

def extract_Resnet50(tensor):
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    return ResNet50(weights='imagenet', include_top=False,pooling="avg").predict(preprocess_input(tensor))

def ResNet50_predict_breed(image_url):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(image_url))
    # obtain predicted vector
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    predicted_vector = ResNet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    df = pd.DataFrame(data={'Prediction': predicted_vector.reshape(-1), 'Dog Breed': dog_names})
    df_sort=df.sort_values(['Prediction'], ascending=False)
    df_sort=df_sort.head(n=5)
    df_sort=df_sort.to_dict('dict')
    df_sort=list(map(lambda x: ({"Prediction": df_sort["Prediction"][x], "Dog Breed": df_sort["Dog Breed"][x]}), list(df_sort["Prediction"].keys())))
    breed = dog_names[np.argmax(predicted_vector)]
    if dog_detector(image_url) == True:
        return df_sort
    else:
        return print("If this person were a dog, the breed would be a {}".format(breed))   

#For prediction it dog or not
def predict_breed(img_path):
    isDog = dog_detector(img_path)
    isPerson = face_detector(img_path)
    if isDog:
        print("Detected a dog")
        breed = ResNet50_predict_breed(img_path)
        return breed
    if isPerson:
        print("Detected a human face")
        breed = ResNet50_predict_breed(img_path)
        return 1
    else:
        print("No human face or dog detected")
        return 1

class predictImage:
    def psy(filepath):

        image_url = tf.keras.utils.get_file(
            origin= "http://127.0.0.1:8000"+filepath
            #origin= "https://dogbreedsapp.herokuapp.com"+filepath
        )
        res=predict_breed(image_url)
        return res


 