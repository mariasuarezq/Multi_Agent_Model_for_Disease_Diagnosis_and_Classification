import os
import json 
import numpy as np
from pathlib import Path
from joblib import load

from utils.functions import get_nlp_prediction, extract_analysis_data_from_text, merge_dictionaries 
from utils.resources import data
import pandas as pd

current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent, 'models')
data_structure = {
        "Glucose": None, "Cholesterol": None, "Hemoglobin": None, "Platelets": None,
         "White Blood Cells": None, "Red Blood Cells": None, "Hematocrit": None, 
         "Mean Corpuscular Volume": None, "Mean Corpuscular Hemoglobin": None, 
         "Mean Corpuscular Hemoglobin Concentration": None, "Insulin": None, "BMI": None, 
         "Systolic Blood Pressure": None, "Diastolic Blood Pressure": None, "Triglycerides": None, 
         "HbA1c": None, "LDL Cholesterol": None, "HDL Cholesterol": None, "ALT": None, "AST": None, 
         "Heart Rate": None, "Creatinine": None, "Troponin": None, "C-reactive Protein": None
         }

def detect_language(state: dict) -> str:
    input_text = state['input_text']
    prediction_mapping = {0: 'English', 1: 'Spanish'}
    language_detected = get_nlp_prediction(input_text=input_text,
                                           prediction_mapping=prediction_mapping,
                                           model_directory='language_detection')
    return {"language_detected": language_detected}


def classify_disease_symptoms_spanish(state: dict) -> str:
    input_text = state['input_text']
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes', 
                          2: 'Talasemia', 
                          3: 'Trombosis'}
    prediction = get_nlp_prediction(input_text=input_text,
                                    prediction_mapping=prediction_mapping,
                                    model_directory='disease_classification_spanish_nlp')
    return {"nlp_disease_prediction": prediction}

def classify_disease_symptoms_english(state: dict) -> str:
    input_text = state['input_text']
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes', 
                          2: 'Thalassemia', 
                          3: 'Thrombosis'}
    prediction = get_nlp_prediction(input_text=input_text,
                                    prediction_mapping=prediction_mapping,
                                    model_directory='disease_classification_english_nlp')
    return {"nlp_disease_prediction": prediction}

def preprocess_data_to_extract(state: dict):
    text_analysis = state['text_analysis']
    return {"text_chunks": [text_analysis]}

def extract_data_from_text_chunks(state: dict):
    text_chunks = state['text_chunks']
    data = data_structure
    for text in text_chunks:
        data_extracted = extract_analysis_data_from_text(text)
        data = merge_dictionaries(data, data_extracted) 
    return {"extracted_data": data}

def classify_disease_from_analysis(state: dict) -> str:
    extracted_data = state['extracted_data']    
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes',
                          2: 'Healthy', 
                          3: 'Thalassemia', 
                          4: 'Thrombosis'}
    model_save_path = os.path.join(MODELS_LOCATION, "disease_classification.joblib")
    xgb_model = load(model_save_path)
    data_df = pd.DataFrame([extracted_data])
    # Predire utilizzando il modello
    pred_encoded = xgb_model.predict(data_df)
    prediction = prediction_mapping[np.argmax(pred_encoded)]
    return {"prediction_based_on_analysis": prediction}
