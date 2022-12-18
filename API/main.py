from typing import Union
import pickle
import os
import glob
import sys

from pydantic import BaseModel
from fastapi import FastAPI
from download_s3_model import download_s3_filemodel

BUCKET_NAME = 'titanic-models-bucket' # replace with your bucket name
KEY = 'models/'

download_s3_filemodel(BUCKET_NAME, KEY)

MODEL_PATH = '/home/ec2-user/environment/api_app/API'

model_list = glob.glob(os.path.join(MODEL_PATH, 'dt_classifier_acc_*'))
acc_list = [int(model.split('/')[-1].replace('dt_classifier_acc_', '')) for model in model_list]
model_path = model_list[acc_list.index(max(acc_list))]
model = pickle.load(open(model_path, 'rb'))

app = FastAPI()

class TictanicData(BaseModel):
    pclass: int
    sex: int
    age: int
    fare: int
    embarked: int
    deck: int
    title: float
    relatives: int
    age_class: int

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def get_prediction(data: TictanicData):
    data = data.dict()
    input_data = [[data['pclass'], data['sex'],
                   data['age'], data['fare'], data['embarked'],
                   data['deck'], data['title'], data['relatives'], 
                   data['age_class']]]
    pred = int(model.predict(input_data)[0])
    if pred:
        return { 'prediction': 'Survived' }

    return { 'prediction': ' Not Survived' }
