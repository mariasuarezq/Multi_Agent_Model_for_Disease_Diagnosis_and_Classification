# Modelo Multiagente para el Diagnóstico y Clasificación de Enfermedades

## Descripción General
Este proyecto es mi Trabajo de Fin de Máster del Máster en Big Data, Data Science e Inteligencia Artificial, cursado en la Universidad Complutense de Madrid. En él, se implementa un modelo multiagente que diagnostica y clasifica enfermedades basándose en los síntomas y datos analíticos proporcionados por los pacientes. El sistema soporta interacción en dos idiomas: inglés y español, ajustándose al idioma hablado por el usuario.

### Funcionamiento del Modelo
1. **Identificación del idioma**: Un modelo NLP detecta si el paciente habla en inglés o en español.
2. **Recopilación de síntomas**: El sistema pregunta al paciente sobre sus síntomas en el idioma detectado y recibe sus respuestas.
3. **Predicción inicial**: Basado en los síntomas proporcionados, se utiliza un modelo NLP para clasificar los síntomas en una de las siguientes enfermedades:
   - Anemia
   - Talasemia
   - Trombosis
   - Diabetes

    Dependiendo del idioma en que el paciente haya proporcionado los síntomas (inglés o español), se utilizará un modelo NLP para clasificar los síntomas en el idioma correspondiente.

4. **Análisis de datos clínicos**: El sistema solicita al paciente que cargue una analítica en uno de los siguientes formatos: CSV, XLSX, TXT, DOCX o PDF. Luego, un modelo clásico de Machine Learning analiza los datos clínicos (como niveles de colesterol, hemoglobina, HbA1c, PCR, etc.) y realiza una predicción más precisa. El resultado puede ser una de las enfermedades mencionadas o indicar que el paciente está sano.

5. **Respuesta final y recomendaciones**: Utilizando un LLM (Large Language Model), el sistema genera un informe detallado que incluye:
   - Diagnóstico final
   - Recomendaciones personalizadas


## Tecnologías Utilizadas
- **Procesamiento de Lenguaje Natural (NLP)**:
  - Modelo para detección de idioma (español/inglés).
  - Modelos de clasificación de síntomas en español e inglés.
- **Machine Learning**:
  - Clasificadores clásicos (SVM, KNN Neighbours, Decission Tree y XGBoost) para análisis de datos clínicos.
- **Large Language Model (LLM)**:
  - Generación de respuestas detalladas y recomendaciones finales mediante el modelo de LLM "Meta-Llama-3.3-70B-Instruct"
- **Integración de formatos**:
  - Procesamiento de archivos CSV, XLSX, TXT, DOCX y PDF para extracción de datos clínicos.


## Guía de Instalación
### Requisitos Previos
- Python 3.8 o superior
- Bibliotecas necesarias (pueden instalarse desde `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Instalación
1. Clona este repositorio:
   ```bash
   git clone https://github.com/mariasuarezq/Multi_Agent_Model_for_Disease_Diagnosis_and_Classification
   ```
2. Accede al directorio del proyecto:
   ```bash
   cd tfm
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso
### Ejecución del Modelo
1. Ejecuta el archivo graph.py
2. Sigue las instrucciones en pantalla:
   - Indica tus síntomas cuando se te soliciten.
   - Sube el archivo con tu analítica cuando el sistema lo requiera.

### Formatos de Entrada
- **Síntomas**: Texto libre en español o inglés.
- **Analítica**: Archivos en formato CSV, XLSX, TXT, DOCX o PDF que contengan datos como niveles de trigicéridos, glucosa, plaquetas, etc.

### Salida del Sistema
- Diagnóstico preliminar basado en síntomas.
- Diagnóstico final basado en datos clínicos.
- Recomendaciones detalladas en el idioma del paciente.



## Licencia
Este proyecto está bajo la licencia [MIT](LICENSE).



  