import io
from PIL import Image
from typing import Any
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph_nodes import (detect_language, 
    classify_disease_symptoms_spanish,
    classify_disease_symptoms_english,
    preprocess_data_to_extract,
    extract_data_from_text_chunks,
    classify_disease_from_analysis)


# Funzione per chiedere l'input manuale dell'utente
def get_user_input(state):
    print("\n Hi i am a virtual assistant designed to detect diseases.")
    print("The disease i can recognize are Anemia, Diabetes, Healthy, Thalassemia and Thrombosis.")
    state["input_text"] = input("Tell me your symptoms, be concise and informative please: \n")
    return state

def get_text_analysis(state):
    print(f"The symptoms may suggest a {state['nlp_disease_prediction']}.")
    print(f"But the symptoms alone are often not sufficient to make an accurate diagnosis.")    
    state["text_analysis"] = input("Could you please insert your blood sample analysis? \n")
    return state


def language_distinction(state):
    language = state["language_detected"]
    return language.lower()


class GraphState(TypedDict):
    """
    Rappresenta lo stato del grafo
    """
    input_text: str
    language_detected: str
    nlp_disease_prediction: str 
    text_analysis: str
    text_chunks: list[str]
    extracted_data: dict[str, Any]
    prediction_based_on_analysis: str


# Initializing the graph
workflow = StateGraph(GraphState)

# Defining the nodes
workflow.add_node("get_user_input", get_user_input)  # Nodo per input iniziale
workflow.add_node("detect_language", detect_language)
workflow.add_node("classify_disease_symptoms_spanish", classify_disease_symptoms_spanish)
workflow.add_node("classify_disease_symptoms_english", classify_disease_symptoms_english)
workflow.add_node("get_text_analysis", get_text_analysis)  # Nodo per input analisi testo
workflow.add_node("preprocess_data_to_extract", preprocess_data_to_extract)
workflow.add_node("extract_data_from_text_chunks", extract_data_from_text_chunks)
workflow.add_node("classify_disease_from_analysis", classify_disease_from_analysis)

# Building the Graph edges
workflow.add_edge(START, "get_user_input")  # Primo input dall'utente
workflow.add_edge("get_user_input", "detect_language")
workflow.add_conditional_edges("detect_language", 
                               language_distinction,
                               {"spanish": "classify_disease_symptoms_spanish", 
                                "english": "classify_disease_symptoms_english"})
workflow.add_edge("classify_disease_symptoms_spanish", "get_text_analysis")
workflow.add_edge("classify_disease_symptoms_english", "get_text_analysis")
workflow.add_edge("get_text_analysis", "preprocess_data_to_extract")  # Input per analisi testo
workflow.add_edge("preprocess_data_to_extract", "extract_data_from_text_chunks")
workflow.add_edge("extract_data_from_text_chunks", "classify_disease_from_analysis")
workflow.add_edge("classify_disease_from_analysis", END)

# compile the graph 
disease_detection_graph = workflow.compile()

# draw the graph 
mermaid_png = disease_detection_graph.get_graph(xray=True).draw_mermaid_png()
image = Image.open(io.BytesIO(mermaid_png))
image.show()

# Importa le dipendenze necessarie se non già importate
from langgraph.graph import END, START, StateGraph

# Usa lo stato iniziale già definito
initial_state = {
    "input_text": "",  
    "language_detected": "",
    "nlp_disease_prediction": "",
    "text_analysis": "",  
    "text_chunks": [],
    "extracted_data": {},
    "prediction_based_on_analysis": ""
}

# Esegui il grafo
result = disease_detection_graph.invoke(initial_state)

# Stampa i risultati
print("\nRisultati dell'analisi:")
print(f"Lingua rilevata: {result['language_detected']}")
print(f"Predizione basata sui sintomi: {result['nlp_disease_prediction']}")
print(f"Predizione finale basata sulle analisi: {result['prediction_based_on_analysis']}")