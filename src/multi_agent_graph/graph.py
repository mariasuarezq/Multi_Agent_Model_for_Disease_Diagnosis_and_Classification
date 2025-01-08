import io
from PIL import Image
from typing import Any
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph_nodes import (detect_language, 
    classify_disease_symptoms_spanish,
    classify_disease_symptoms_english,
    preprocess_data_to_extract,
    extract_analysis_data_from_text,
    classify_disease_from_analysis)

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
workflow.add_node("detect_language", detect_language)
workflow.add_node("classify_disease_symptoms_spanish", classify_disease_symptoms_spanish)
workflow.add_node("classify_disease_symptoms_english", classify_disease_symptoms_english)
workflow.add_node("preprocess_data_to_extract", preprocess_data_to_extract)
workflow.add_node("extract_analysis_data_from_text", extract_analysis_data_from_text)
workflow.add_node("classify_disease_from_analysis", classify_disease_from_analysis)

# Building the Graph edges
workflow.add_edge(START, "detect_language")
workflow.add_conditional_edges("detect_language", 
                               language_distinction,
                               {"spanish": "classify_disease_symptoms_spanish", 
                                "english": "classify_disease_symptoms_english"})
workflow.add_edge("classify_disease_symptoms_spanish", "preprocess_data_to_extract")
workflow.add_edge("classify_disease_symptoms_english", "preprocess_data_to_extract")
workflow.add_edge("preprocess_data_to_extract", "extract_analysis_data_from_text")
workflow.add_edge("extract_analysis_data_from_text", "classify_disease_from_analysis")
workflow.add_edge("classify_disease_from_analysis", END)


# compile the graph 
disease_detection_graph = workflow.compile()


# draw the graph 
mermaid_png = disease_detection_graph.get_graph(xray=True).draw_mermaid_png()
image = Image.open(io.BytesIO(mermaid_png))
image.show()


