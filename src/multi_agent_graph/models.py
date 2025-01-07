import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent, 'models')


def detect_language(input_text: str) -> str:
    prediction_mapping = {0: 'English', 1: 'Spanish'}
    model_directory = os.path.join(MODELS_LOCATION, 'language_detection')
    # Carica il tokenizer e il modello
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory, local_files_only=True)
    # Tokenizza la frase
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    # Effettua la predizione
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, axis=1).item()
    return prediction_mapping[prediction]


def classify_disease_symptoms(input_text: str) -> str:
    prediction_mapping = {0: 'Anemia',
                          1: 'Diabetes', 
                          2: 'Talasemia', 
                          3: 'Trombosis'}
    model_directory = os.path.join(MODELS_LOCATION, 'disease_classification_spanish_nlp')
    # Carica il tokenizer e il modello
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory, local_files_only=True)
    # Tokenizza la frase
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    # Effettua la predizione
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, axis=1).item()
    return prediction_mapping[prediction]
    


response = classify_disease_symptoms(
    "Tengo dolores en los huesos y me siento agotado incluso después de descansar.")
print(response)