from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import keras.utils as image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model
from numpy import asarray
from PIL import Image

import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms

model = tf.keras.models.load_model('models/mymodel.h5')

model_torch = torchvision.models.resnet50(pretrained=True)
model_torch.fc = nn.Sequential(nn.Linear(2048, 46))
model_torch.load_state_dict(torch.load('models/retinal_disease.pt',map_location=torch.device('cpu')))

def makepred(object):
    pil_image = Image.open(object).convert('RGB') 
    img=np.array(pil_image)
    img.resize((150,150,3))
    img = np.expand_dims(img, axis=0)
    img=img/255.0
    pred=model.predict(img)
    pred=pred > 0.35809565
    return pred[0][0]

def pre_image(obj,model):
    img = Image.open(obj)
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((1424, 2144)),transforms.Normalize(mean, std)])
    # get normalized image
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    # input = Variable(image_tensor)
    # print(img_normalized.shape)
    with torch.no_grad():
        model.eval()  
        output =model(img_normalized)
        output=torch.sigmoid(output[0])
        # print(output)
        return str(np.array((output > 0.5)[0])), str(np.array((output[0])))


app = Flask(__name__)
CORS(app)

@app.route('/make_prediction/', methods=['GET', 'POST'])
def welcome():
    try:
        f = request.files['file']
        otpt_tf=makepred(f)
        otpt_torch=pre_image(f,model_torch)
        otpt_dict={'tf_prediction':str(otpt_tf),'torch_prediction':str(otpt_torch[0]),'probability':otpt_torch[1]}
        return jsonify(otpt_dict),200
    except Exception as e:
        otpt_dict={'tf_prediction':str(False),'torch_prediction':str(False),'probability':0.4201}
        print("exception caused :",str(e))
        return jsonify(otpt_dict),200

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug = True)


#/home/daniyal/anaconda3/envs/rtc