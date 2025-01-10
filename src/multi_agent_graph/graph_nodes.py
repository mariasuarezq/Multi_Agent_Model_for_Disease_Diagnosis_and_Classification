import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from joblib import load

from utils.functions import (get_nlp_prediction, 
                             extract_data_from_text_chunks, 
                             preprocess_data_to_extract,
                             get_text_analysis,
                             get_user_input,
                             generate_final_answer)

warnings.filterwarnings("ignore")
current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent, 'models')

def detect_language(state: dict) -> str:
    input_text = get_user_input()
    prediction_mapping = {0: 'English', 1: 'Spanish'}
    detected_language = get_nlp_prediction(input_text=input_text,
                                           prediction_mapping=prediction_mapping,
                                           model_directory='language_detection')
    return {"detected_language": detected_language, "input_text": input_text}


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

def classify_disease_from_analysis(state: dict) -> str:
    text_blood_sample_analysis = get_text_analysis(state['detected_language'],
                                                   state['nlp_disease_prediction'])
    text_chunks = preprocess_data_to_extract(text_blood_sample_analysis)
    extracted_data = extract_data_from_text_chunks(text_chunks)
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

def final_asnwer_generator(state: dict) -> str:
    final_answer = generate_final_answer(
        state["nlp_disease_prediction"],
        state["prediction_based_on_analysis"],
        state["detected_language"]
    )
    print(f"\n\n\n\n {final_answer} \n\n")
    return {"final_answer": final_answer}