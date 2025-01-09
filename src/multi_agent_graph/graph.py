import io
import os
import warnings
from PIL import Image
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph_nodes import (
    detect_language, 
    classify_disease_symptoms_spanish,
    classify_disease_symptoms_english,
    classify_disease_from_analysis,
    final_asnwer_generator
    )


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=DeprecationWarning)  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def language_distinction(state):
    language = state["detected_language"]
    return language.lower()


class GraphState(TypedDict):
    """
    Rappresenta lo stato del grafo
    """
    input_text: str
    detected_language: str
    nlp_disease_prediction: str 
    prediction_based_on_analysis: str
    final_answer: str


# Initializing the graph
workflow = StateGraph(GraphState)

# Defining the nodes
workflow.add_node("detect_language", detect_language)
workflow.add_node("classify_disease_symptoms_spanish", classify_disease_symptoms_spanish)
workflow.add_node("classify_disease_symptoms_english", classify_disease_symptoms_english)
workflow.add_node("classify_disease_from_analysis", classify_disease_from_analysis)
workflow.add_node("final_asnwer_generator", final_asnwer_generator)

# Building the Graph edges
workflow.add_edge(START, "detect_language")
workflow.add_conditional_edges("detect_language", 
                               language_distinction,
                               {"spanish": "classify_disease_symptoms_spanish", 
                                "english": "classify_disease_symptoms_english"})
workflow.add_edge("classify_disease_symptoms_spanish", "classify_disease_from_analysis")
workflow.add_edge("classify_disease_symptoms_english", "classify_disease_from_analysis")
workflow.add_edge("classify_disease_from_analysis", "final_asnwer_generator")
workflow.add_edge("final_asnwer_generator", END)

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
    "detected_language": "",
    "nlp_disease_prediction": "",
    "text_analysis": "",  
    "text_chunks": [],
    "extracted_data": {},
    "prediction_based_on_analysis": ""
}

# Esegui il grafo
result = disease_detection_graph.invoke(initial_state)