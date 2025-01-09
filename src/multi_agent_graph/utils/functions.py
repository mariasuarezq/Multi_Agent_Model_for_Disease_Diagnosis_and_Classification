import os
import json
import torch
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent.parent, 'models')
CONFIG_LOCATION = os.path.join(current_dir.parent.parent, 'config.json')
PROMPTS_LOCATION = os.path.join(current_dir.parent, 'utils', 'prompts')
8/3
data_structure = {
        "Glucose": None, "Cholesterol": None, "Hemoglobin": None, "Platelets": None,
         "White Blood Cells": None, "Red Blood Cells": None, "Hematocrit": None, 
         "Mean Corpuscular Volume": None, "Mean Corpuscular Hemoglobin": None, 
         "Mean Corpuscular Hemoglobin Concentration": None, "Insulin": None, "BMI": None, 
         "Systolic Blood Pressure": None, "Diastolic Blood Pressure": None, "Triglycerides": None, 
         "HbA1c": None, "LDL Cholesterol": None, "HDL Cholesterol": None, "ALT": None, "AST": None, 
         "Heart Rate": None, "Creatinine": None, "Troponin": None, "C-reactive Protein": None
         }

with open(CONFIG_LOCATION, 'r') as configs:
    config = json.load(configs)
samba_cloud_api_key = config['samba_cloud_api_key']
client = OpenAI(
    api_key=samba_cloud_api_key,
    base_url="https://api.sambanova.ai/v1/"
)


def get_nlp_prediction(input_text: str, prediction_mapping: dict, model_directory: str) -> str:
    model_path = os.path.join(MODELS_LOCATION, model_directory)
    # Carica il tokenizer e il modello
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    # Tokenizza la frase
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    # Effettua la predizione
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, axis=1).item()
    return prediction_mapping[prediction]

def merge_dictionaries(dict1: dict, dict2: dict):
    """
    copia i valori non None da dict1 e aggiunge i valori non None da dict2 solo quando la chiave 
    corrispondente in dict1 è None.
    """
    merged_dict = {}
    for key in dict1:
        if dict1[key] is not None:
            merged_dict[key] = dict1[key]
        elif dict2[key] is not None:
            merged_dict[key] = dict2[key]

    return merged_dict

def extract_analysis_data_from_text(text):
    model = 'Meta-Llama-3.3-70B-Instruct'
    with open(os.path.join(PROMPTS_LOCATION, 'analysis_data_extraction.txt'), "r") as file:
        prompt_data_extraction = file.read()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_data_extraction},
            {"role": "user", "content": text}
        ],
        stream=False
    )
    response = completion.choices[0].message.content.lstrip('json').lstrip('`').rstrip('`').lstrip('json')
    return json.loads(response)

def generate_final_answer(symptoms_prediction: str, analysis_prediction: str, detected_language: str):
    model = 'Meta-Llama-3.3-70B-Instruct'
    user_prompt = f""" This is the prediction based on the symptoms: {symptoms_prediction}.
                    This is the prediction based on the analysis: {analysis_prediction}.
                    Answer in {detected_language}. """
    with open(os.path.join(PROMPTS_LOCATION, 'final_answer_generation.txt'), "r") as file:
        prompt_final_answer_generation = file.read()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_final_answer_generation},
            {"role": "user", "content": user_prompt}
        ],
        stream=False
    )
    response = completion.choices[0].message.content
    return response

def get_text_analysis(detected_language: str, nlp_disease_prediction: str):
    if detected_language.lower()=="spanish":
        print(f"Los síntomas pueden sugerir un {nlp_disease_prediction}.")
        print(f"Pero los síntomas por sí solos a menudo no son suficientes para realizar un diagnóstico preciso.")    
        return input("¿Podrías por favor insertar el análisis de tu muestra de sangre? \n")

    else:
        print(f"The symptoms may suggest a {nlp_disease_prediction}.")
        print(f"But the symptoms alone are often not sufficient to make an accurate diagnosis.")    
        return input("Could you please insert your blood sample analysis? \n")

def preprocess_data_to_extract(text_analysis: str):
    return [text_analysis]

def extract_data_from_text_chunks(text_chunks: list):
    data = data_structure
    for text in text_chunks:
        data_extracted = extract_analysis_data_from_text(text)
        data = merge_dictionaries(data, data_extracted) 
    return data

def get_user_input():
    print("\n Hi i am a virtual assistant designed to detect diseases.")
    print("The disease i can recognize are Anemia, Diabetes, Healthy, Thalassemia and Thrombosis.")
    return input("Tell me your symptoms, be concise and informative please: \n")