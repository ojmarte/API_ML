from typing import Union
import pickle
import os
import glob
import sys
sys.path.append('../titanic_challenge/src')

from pydantic import BaseModel
from pipeline import create_preprocessing_pipeline, create_feature_engineering_pipeline, create_ml_pipeline, prepare_submission, create_single_preprocessing_pipeline

from fastapi import FastAPI

MODEL_PATH = '/Users/alexanderholguin/MLPython/titanic_challenge/data'

model_list = glob.glob(os.path.join(MODEL_PATH, 'dt_classifier_acc_*'))
acc_list = [int(model.split('/')[-1].replace('dt_classifier_acc_', '')) for model in model_list]
model_path = model_list[acc_list.index(max(acc_list))]
model = pickle.load(open(model_path, 'rb'))

app = FastAPI()

# def run_ml_model(data: dict):
#     train_path = os.path.join(path, 'train.csv')
#     test_path = os.path.join(path, 'test.csv')
#     submission_path = os.path.join(path, 'submission.csv')

#     train_df = create_single_preprocessing_pipeline(data, True)
#     features_df = create_feature_engineering_pipeline(train_df)
#     model, training_acc = create_ml_pipeline(features_df)
#     pickle.dump(model, open(f'dt_classifier_acc_{round(training_acc)}', 'wb'))

#     return prepare_submission(model, test_path, submission_path)

# train.csv
# test.csv
# submission.csv
# PassengerId  Pclass  Sex  Age  Fare  Embarked  Deck  Title  Relatives  Age_Class
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
