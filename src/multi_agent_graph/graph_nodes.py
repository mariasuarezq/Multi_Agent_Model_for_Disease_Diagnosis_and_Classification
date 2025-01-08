import os
import json 
import numpy as np
from pathlib import Path
from joblib import load

from utils.functions import get_nlp_prediction, extract_analysis_data_from_text, merge_dictionaries 
from utils.resources import data
import pandas as pd

current_dir = Path(__file__).parent
CONFIG_LOCATION = os.path.join(current_dir.parent.parent, 'config.json')
MODELS_LOCATION = os.path.join(current_dir.parent, 'models')


def detect_language(input_text: str) -> str:

    prediction_mapping = {0: 'English', 1: 'Spanish'}
    language_detected = get_nlp_prediction(input_text=input_text,
                                           prediction_mapping=prediction_mapping,
                                           model_directory='language_detection')
    return {"language_detected": language_detected}


def classify_disease_symptoms_spanish(input_text: str) -> str:
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes', 
                          2: 'Talasemia', 
                          3: 'Trombosis'}
    prediction = get_nlp_prediction(input_text=input_text,
                                    prediction_mapping=prediction_mapping,
                                    model_directory='disease_classification_spanish_nlp')
    return {"nlp_disease_prediction": prediction}

def classify_disease_symptoms_english(input_text: str) -> str:
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes', 
                          2: 'Thalassemia', 
                          3: 'Thrombosis'}
    prediction = get_nlp_prediction(input_text=input_text,
                                    prediction_mapping=prediction_mapping,
                                    model_directory='disease_classification_english_nlp')
    return {"nlp_disease_prediction": prediction}

def preprocess_data_to_extract(text_analysis):
    return [text_analysis]

def extract_data_from_text_chunks(text_chunks: list[str]):
    for text in text_chunks:
        data_extracted = extract_analysis_data_from_text(text)
        data = merge_dictionaries(data, data_extracted) 
    return {"extracted_data": data}

def classify_disease_from_analysis(extracted_data: dict) -> str:
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
