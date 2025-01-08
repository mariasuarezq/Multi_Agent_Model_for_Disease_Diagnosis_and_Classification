import os
import json
import torch
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent, 'models')
CONFIG_LOCATION = os.path.join(current_dir.parent.parent, 'config.json')
PROMPTS_LOCATION = os.path.join(current_dir.parent, 'utils', 'prompts')

with open(CONFIG_LOCATION, 'r') as configs:
    config = json.load(configs)
with open(os.path.join(PROMPTS_LOCATION, 'analysis_data_extraction.txt'), "r") as file:
    prompt_data_extraction = file.read()
samba_cloud_api_key = config['samba_cloud_api_key']


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
    client = OpenAI(
        api_key=samba_cloud_api_key,
        base_url="https://api.sambanova.ai/v1/"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_data_extraction},
            {"role": "user", "content": text}
        ],
        stream=False
    )
    response = completion.choices[0].message.content.lstrip('`').rstrip('`')
    return json.loads(response)
