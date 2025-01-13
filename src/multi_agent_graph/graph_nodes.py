import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from joblib import load

from utils.functions import (get_nlp_prediction, 
                             extract_data_from_text_chunks, 
                             preprocess_data_to_extract,
                             get_analysis_doc_name,
                             get_user_input,
                             generate_final_answer)

warnings.filterwarnings("ignore")
current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent, 'models')


def detect_language(state: dict) -> str:
    """
    Esta función detecta el idioma del texto proporcionado por el usuario.
    Llama a la función get_user_input() para obtener el texto y luego utiliza 
    get_nlp_prediction() para hacer la predicción sobre el idioma del texto.
    Devuelve un diccionario con el idioma detectado y el texto introducido.
    """
    input_text = get_user_input()
    prediction_mapping = {0: 'English', 1: 'Spanish'}
    detected_language = get_nlp_prediction(input_text=input_text,
                                           prediction_mapping=prediction_mapping,
                                           model_directory='language_detection')
    return {"detected_language": detected_language, "input_text": input_text}


def classify_disease_symptoms_spanish(state: dict) -> str:
    """
    Esta función clasifica los síntomas proporcionados por el usuario en una de las enfermedades predefinidas:
    Anemia, Diabetes, Talasemia o Trombosis, usando un modelo de clasificación de enfermedades en español.
    Toma el texto de los síntomas desde el estado, y luego usa la función get_nlp_prediction() para hacer la predicción.
    Devuelve un diccionario con la predicción de la enfermedad.
    """
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
    """
    Esta función clasifica los síntomas proporcionados por el usuario en una de las enfermedades predefinidas:
    Anemia, Diabetes, Thalassemia o Thrombosis, usando un modelo de clasificación de enfermedades en inglés.
    Toma el texto de los síntomas desde el estado, y luego usa la función get_nlp_prediction() para hacer la predicción.
    Devuelve un diccionario con la predicción de la enfermedad.
    """
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
    """
    Esta función clasifica la enfermedad basándose en un análisis de sangre. 
    Primero, obtiene el nombre del documento del análisis utilizando la función 'get_analysis_doc_name',
    luego procesa el texto usando la función 'preprocess_data_to_extract', 
    extrae los datos con 'extract_data_from_text_chunks', 
    y finalmente realiza una predicción utilizando el modelo cargado desde un archivo con 'xgb_model.predict'.
    """
    blood_analysis_doc_name = get_analysis_doc_name(state['detected_language'],
                                                    state['nlp_disease_prediction']).lower()
    text_chunks = preprocess_data_to_extract(blood_analysis_doc_name)
    extracted_data = extract_data_from_text_chunks(text_chunks)
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes',
                          2: 'Healthy', 
                          3: 'Thalassemia', 
                          4: 'Thrombosis'}
    model_save_path = os.path.join(MODELS_LOCATION, "disease_classification.joblib")
    xgb_model = load(model_save_path)
    data_df = pd.DataFrame([extracted_data])
    # Predecir utilizando el modelo
    pred_encoded = xgb_model.predict(data_df)
    prediction = prediction_mapping[np.argmax(pred_encoded)]
    return {"prediction_based_on_analysis": prediction}

def final_answer_generator(state: dict) -> str:
    """
    Esta función genera la respuesta final basándose en la predicción de los síntomas y el análisis.
    Primero, llama a la función generate_final_answer pasando la predicción de los síntomas, 
    la predicción basada en el análisis de sangre y el idioma detectado. Luego, imprime y devuelve 
    la respuesta generada, para finalmente utilizar la función generate_final_answer para elaborar 
    la respuesta final.
    """
    final_answer = generate_final_answer(
        state["nlp_disease_prediction"],
        state["prediction_based_on_analysis"],
        state["detected_language"]
    )
    print(f"\n\n\n\n {final_answer} \n\n")
    return {"final_answer": final_answer}