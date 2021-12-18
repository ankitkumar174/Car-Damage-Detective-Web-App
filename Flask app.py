import os
from flask import Flask, render_template, request
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd

app=Flask(__name__)

model_0 = load_model('car damage or not detection model.h5')
model_1 = load_model('car damage location model.h5', compile=False)
model_2 = load_model('car casuality level model.h5', compile=False)

@app.route('/', methods=['GET'])
def home():
    return render_template('degign.html')

@app.route('/',methods=["POST"])

# Process file and predict his label
def predict():
    imgfile=request.files['image']
    image_path =  "./static/" + imgfile.filename
    imgfile.save(image_path)

    img=cv2.imread(image_path)
    img = cv2.resize(img,(224,224))
    img = img /255.
    img = img.reshape(224,224,3)
    img=tf.expand_dims(img,axis=0)

    
    
    pred1=model_0.predict(img)
    pred2=model_1.predict(img)
    pred3=model_2.predict(img)
    
    pred1=pred1.round()
    if pred1==0:
        pred="Damaged"
    else:
        pred="Not Damaged"

    pred2=pred2.argmax()
    if pred2==0:
        location='Front'
    elif pred2==1:
        location='Rear'
    else:
        location='Side'
    

    pred3=pred3.argmax()
    if pred3==0:
        severe='Minor'
    elif pred3==1:
        severe='Moderate'
    else:
        severe='Critical'
        
    if pred1==0:
        return render_template('degign.html', damage=pred, location=location, severity=severe, final="Damage Detection Successfull!", image_loc=imgfile.filename)
    else:
        return render_template('degign.html', damage=pred, location="Not Applicable", severity="Not Applicable", final="Your Car is Not Damaged", info="If you found any discrepancy with above result. Please try again with another image.", image_loc=imgfile.filename)

if '__main__'==__name__:
    app.run(debug=True)
