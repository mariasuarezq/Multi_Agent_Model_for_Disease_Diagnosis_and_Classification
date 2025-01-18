# Modelo Multiagente para el Diagnóstico y Clasificación de Enfermedades

## Descripción General
Este proyecto implementa un modelo multiagente que diagnostica y clasifica enfermedades basándose en los síntomas y datos analíticos proporcionados por los pacientes. El sistema soporta interacción en dos idiomas: inglés y español, ajustándose al idioma hablado por el usuario.

## Funcionamiento del Modelo
1. **Identificación del idioma**: Un modelo NLP detecta si el paciente habla en inglés o en español.
2. **Recopilación de síntomas**: El sistema pregunta al paciente sobre sus síntomas en el idioma detectado y recibe sus respuestas.
3. **Predicción inicial**: Basado en los síntomas proporcionados, se utiliza un modelo NLP para clasificar los síntomas en una de las siguientes enfermedades:
   - Anemia
   - Talasemia
   - Trombosis
   - Diabetes

    Dependiendo del idioma en que el paciente haya proporcionado los síntomas (inglés o español), se utilizará un modelo NLP para clasificar los síntomas en el idioma correspondiente.
4. **Análisis de datos clínicos**: El sistema solicita al paciente que cargue una analítica en uno de los siguientes formatos: CSV, XLSX, TXT, DOCX o PDF. Luego, un modelo clásico de Machine Learning analiza los datos clínicos (glucosa (mg/dL), colesterol (mg/dL), hemoglobina (g/L), plaquetas (por microlitro de sangre), glóbulos blancos (por milímetros cúbicos de sangre), glóbulos rojos (millones de células por microlitros de sangre), hematocritos (porcentaje), volumen corpuscular medio (VCM) (femtolitro), hemoglobina corpuscular media (HCM) (picogramos), concentración de hemoglobina corpurscular media (CHCM) (gramos por decilitro), insulina (microU/mL), índice de masa corporal (IMC) (kg/m^2), presión arterial sistólica (mmHg), presión arterial diastólica (mmHg), triglicéridos (mg/dL), HbA1c (hemoglobina glicosilada) (porcentaje), colesterol LDL (mg/dL), colesterol HDL (mg/dL), alanina aminotransferasa (ALT) (U/L), aspartato aminotransferasa (AST) (U/L), frecuencia cardiaca (latidos por minuto), creatinina (mg/dL), troponina (ng/mL) y proteína C reactiva (PCR) (mg/L)) y realiza una predicción más precisa. Estos datos ya están escalados en el rango (0,1). El resultado puede ser una de las enfermedades mencionadas o indicar que el paciente está sano.

5. **Respuesta final y recomendaciones**: Utilizando un LLM (Large Language Model), el sistema genera un informe detallado que incluye:
   - Diagnóstico final
   - Recomendaciones personalizadas

   ![Texto alternativo](src\multi_agent_graph\arquitectura_modelo.jpg)

## Tecnologías Utilizadas
- **Procesamiento de Lenguaje Natural (NLP)**:
  - Modelo para detección de idioma (español/inglés). Hemos utilizado el modelo preentrenado distilbert para el fine tuning.
  - Modelo de clasificación de síntomas en español, donde se ha utilizado el modelo preentrenado bertin para el fine tuning.
  - Modelo de clasificación de síntomas en inglés, donde se ha utilizado el modelo preentrenado distilbert para el fine tuning.
- **Machine Learning**:
  - Clasificadores clásicos (SVM, KNN Neighbours, Decission Tree y XGBoost) para análisis de datos clínicos, eligiendo el modelo construido con XGBoost como el mejor en términos de robustez y generalización.
- **Large Language Model (LLM)**:
  - Generación de respuestas detalladas y recomendaciones finales mediante el modelo de LLM "Meta-Llama-3.3-70B-Instruct"
- **Integración de formatos**:
  - Procesamiento de archivos CSV, XLSX, TXT, DOCX y PDF para extracción de datos clínicos.


## Guía de Instalación
### Requisitos Previos
- Python 3.11.7

### Instalación
1. Clona este repositorio.
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso
### Pasos Preliminares
1. Ejecuta los siguientes archivos, que se encuentran dentro de la carpeta "models_training", descomentando la celda donde se guardan los modelos. Estos modelos se guardarán en la carpeta "models":
   - disease_classification.ipynb
   - language_detection.ipynb
   - nlp_disease_classification_en.ipynb
   - nlp_disease_classification_sp.ipynb
2. Desde la web https://cloud.sambanova.ai/, genera una api key para el modelo Meta-Llama-3.3-70B-Instruct. Guárdala en un archivo llamado config.json, en la root del proyecto, en este formato:

```json
{
  "samba_cloud_api_key": "api_key"
}
   ```
3. Carga tu analítica en la carpeta src\multi_agent_graph\utils\analysis_documents.
### Ejecución del Modelo
1. Ejecuta el archivo graph.py
2. Sigue las instrucciones en pantalla:
   - Indica tus síntomas cuando se te soliciten.
   - Escribe el nombre del archivo (incluyendo la extensión) que contiene tu analítica cuando el sistema lo requiera.

### Formatos de Entrada
- **Síntomas**: Texto libre en español o inglés.
- **Analítica**: Archivos en formato CSV, XLSX, TXT, DOCX o PDF que contengan datos como niveles de trigicéridos, glucosa, plaquetas, etc.

### Salida del Sistema
- Diagnóstico preliminar basado en síntomas.
- Diagnóstico final basado en datos clínicos.
- Recomendaciones detalladas en el idioma del paciente.



  