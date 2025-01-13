import os
import json
import torch
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

current_dir = Path(__file__).parent
MODELS_LOCATION = os.path.join(current_dir.parent.parent, 'models')
CONFIG_LOCATION = os.path.join(current_dir.parent.parent, 'config.json')
PROMPTS_LOCATION = os.path.join(current_dir.parent, 'utils', 'prompts')
DOCUMENTS_LOCATION = os.path.join(current_dir.parent, 'utils', 'analysis_documents')

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


def get_user_input():
    """
    Esta función imprime un mensaje explicativo y luego solicita al usuario que introduzca sus síntomas.
    La entrada del usuario es devuelta como una cadena de texto.
    """
    print("\n Hi i am a virtual assistant designed to detect diseases.")
    print("The disease i can recognize are Anemia, Diabetes, Healthy, Thalassemia and Thrombosis.")
    return input("Tell me your symptoms, be concise and informative please: \n")

def get_nlp_prediction(input_text: str, prediction_mapping: dict, model_directory: str) -> str:
    """
    Esta función usa un modelo de NLP para predecir el idioma del texto proporcionado.
    Carga el modelo y el tokenizer desde una ruta local, tokeniza el texto, hace la predicción y luego mapea 
    el resultado de la predicción al idioma correspondiente usando el diccionario de predicciones proporcionado.
    """
    model_path = os.path.join(MODELS_LOCATION, model_directory)
    # Carga el modelo
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    # Tokeniza la frase
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    # Efectua la prediccion
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, axis=1).item()
    return prediction_mapping[prediction]

def get_analysis_doc_name(detected_language: str, nlp_disease_prediction: str):
    """
    Esta función solicita al usuario el nombre del documento del análisis de sangre. 
    El texto de la solicitud se personaliza según el idioma detectado y la enfermedad sugerida por el análisis previo.
    """
    if detected_language.lower()=="spanish":
        print(f"Los síntomas pueden sugerir un {nlp_disease_prediction}.")
        print(f"Pero los síntomas por sí solos a menudo no son suficientes para realizar un diagnóstico preciso.")  
        print("¿Podrías por favor insertar el nombre del documento con las análisis de tu muestra de sangre?")  
        return input("Los formatos aceptados son txt, csv, xlsx, docx and pdf: \n")

    else:
        print(f"The symptoms may suggest a {nlp_disease_prediction}.")
        print(f"But the symptoms alone are often not sufficient to make an accurate diagnosis.")  
        print("Could you please insert the document's name with your blood sample analysis?")
        return input("The supported extensions are son txt, csv, xlsx, docx and pdf: \n")

def find_and_read_document(filename):
    """
    Esta función busca el documento proporcionado en el directorio de documentos. Si no se proporciona una 
    extensión, intenta encontrar un archivo con cualquier extensión válida (.txt, .csv, .xlsx, .docx, .pdf).
    Luego lee el contenido del archivo en función de su extensión (txt, csv, xlsx, pdf, docx).
    """
    # Obtenemos la lista de todos los archivos en el directorio
    files_in_dir = os.listdir(DOCUMENTS_LOCATION)
    
    # Verificamos si el nombre de archivo proporcionado tiene una extensión
    if "." in filename:
        file_to_find = filename
    else:
        # Buscamos el archivo con el nombre dado y cualquier extensión válida
        valid_extensions = ['.txt', '.csv', '.xlsx', '.docx', '.pdf']
        file_to_find = next((file for file in files_in_dir if file.startswith(filename)), None)
        if os.path.splitext(file_to_find)[1] not in valid_extensions:
            raise ValueError("Document extension not supported.")
    
    # Si encontramos el archivo, determinamos su tipo y lo leemos
    if file_to_find:
        file_path = os.path.join(DOCUMENTS_LOCATION, file_to_find)
        _, file_extension = os.path.splitext(file_to_find)

        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content

        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
            if len(df) != 1:
                raise ValueError("The csv file must have one row with the analysis result.")
            result_text = ". ".join([f"{col} value: {df.iloc[0][col]}" for col in df.columns])
            return result_text

        elif file_extension == '.xlsx':
            df = pd.read_excel(file_path)
            if len(df) != 1:
                raise ValueError("The excel file must only one row with the analysis result.")
            result_text = ". ".join([f"{col} value: {df.iloc[0][col]}" for col in df.columns])
            return result_text
        
        elif file_extension == '.pdf':
            pdf_reader = PdfReader(file_path)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content.strip()
        
        elif file_extension == '.docx':
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content.strip()

        else:
            raise ValueError("Unsupported file type.")
    else:
        raise ValueError(f"File '{filename}' not found.")
    
def preprocess_data_to_extract(file_name: str):
    """
    Esta función procesa el nombre del archivo recibido, encuentra el documento correspondiente utilizando
    la función find_and_read_document y luego divide el contenido en fragmentos de texto para su posterior análisis
    usando 'RecursiveCharacterTextSplitter'.
    """
    text_analysis = find_and_read_document(file_name)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".\n", ". ", "."],  
        chunk_size=2000,
        chunk_overlap=100
    )   
    text_chunks = text_splitter.split_text(text_analysis)
    return text_chunks

def extract_data_from_text_chunks(text_chunks: list):
    """
    Esta función toma una lista de fragmentos de texto y extrae y fusiona los datos relevantes. 
    Para extraer los datos, se llama a la función extract_analysis_data_from_text para cada fragmento,
    y luego se fusionan los datos utilizando la función 'merge_dictionaries'.
    Si faltan valores, se genera un error indicando qué datos faltan.
    """
    data = data_structure
    for text in text_chunks:
        data_extracted = extract_analysis_data_from_text(text)
        data = merge_dictionaries(data, data_extracted) 

    missing_values = [key for key, value in data.items() if value is None]
    if missing_values:
        error_log = "\n".join(missing_values)
        raise ValueError(f"The model couldn't extract the data for the following values: \n {error_log}")
    else:
        return data
    
def extract_analysis_data_from_text(text):
    """
    Esta función utiliza el modelo de LLM 'Meta-Llama-3.3-70B-Instruct' para extraer los datos 
    relevantes de un texto de análisis de sangre, usando un prompt almacenado en un archivo externo. 
    La función se comunica con un servicio de completado de lenguaje para realizar esta extracción.
    """
    model = 'Meta-Llama-3.3-70B-Instruct'
    with open(os.path.join(PROMPTS_LOCATION, 'analysis_data_extraction.txt'), "r") as file:
        prompt_data_extraction = file.read()
    try:
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
    except Exception as e:
        print(f"Error during data extraction from text: {e}")
        return data_structure
    
def merge_dictionaries(dict1: dict, dict2: dict):
    """
    Esta función fusiona dos diccionarios. Si una clave en el primer diccionario tiene un valor None, 
    se reemplaza con el valor correspondiente del segundo diccionario.
    """
    for key in dict1:
        if dict1[key] is None:
            dict1[key] = dict2[key]
    return dict1


def generate_final_answer(symptoms_prediction: str, analysis_prediction: str, detected_language: str):
    """
    Esta función genera la respuesta final que combina las predicciones de los síntomas y del análisis 
    basándose en un modelo LLM llamado Meta-Llama.
    Se forma un prompt que incluye ambas predicciones y el idioma detectado, y luego se genera una respuesta 
    utilizando el modelo de lenguaje.
    """
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