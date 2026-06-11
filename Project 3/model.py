import langchain
import langgraph
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START,StateGraph,END
from IPython.display import display,Image
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
from datetime import datetime

model = ChatGroq(model="llama-3.1-8b-instant")
tavily = TavilySearchResults(max_results=3)
tool = [tavily]
llm = model.bind_tools(tool)

class MedicalState(TypedDict):
    collect_input: str
    extract_symptoms: str
    emergency: bool
    get_location: str
    hospitals: str
    disease_predictor: str
    specialist_recommender: str
    final_report: str


def extract_symptoms(state:MedicalState):
    "Extract symptoms from the user data."
    message = [
        SystemMessage(content=f"You are a Medical AI Assistant."),
        HumanMessage(content=f"Find only the Symptoms from the input {state['collect_input']}")
    ]

    result = model.invoke(message)
    return {"extract_symptoms":result.content}

def emergency(state: MedicalState):

    message = [
        HumanMessage(
            content=f"""
            Symptoms: {state['extract_symptoms']}

            Determine whether this is a medical emergency.

            Return ONLY:
            True
            or
            False
            """
        )
    ]

    result = model.invoke(message)

    value = result.content.strip().lower()

    return {
        "emergency": value == "true"
    }

def get_location(state:MedicalState):
    message = [
        HumanMessage(content=f"Extract the address of the user from {state['collect_input']} and return only address.")
    ]
    result = model.invoke(message)
    return {"get_location":result.content}

def hospital_finder(state:MedicalState):
#     message = [ SystemMessage(content=f"""You are an AI Assistant if you need date and time use it {datetime.now()} and return the best answer.
#                               if needed use tools for best result in order to avoid knowledge cut off date,Use Tavily search to find actual hospitals.
# Return hospital name, address and contact number."""),
#         HumanMessage(content=f"""
#             Based on the symptoms {state['extract_symptoms']} and and the urgency from {state["emergency"]}
#             List the nearest hospital using the address {state["get_location"]}
#                 """)
#     ]
    # response = llm.invoke(message)

    # if response.content:
    #     return {"hospitals": response.content}

    # return {"hospitals": "No hospitals found."}

    results = tavily.invoke(
    {
        "query": f"Top emergency hospitals near {state['get_location']}"
    }
)

    return {"hospitals": results}

def disease(state:MedicalState):
    message = [
        HumanMessage(content=f"""
            Based on the Sympotms {state['extract_symptoms']}.
            Predict the User suffering from which disease with confidence score out of 100 in percentage.
            Also write two to five Facts on ground of which you will say suffering to this disease.
""")
    ]
    result = model.invoke(message)
    return {"disease_predictor":result.content}

def specialist(state:MedicalState):
    message = [
        HumanMessage(content=f"""
        Keeping the symptoms and Diseases tell to which specialist i need to connect. for example:
                     Physicia , Neurologist etc.
                     symptoms :{state["extract_symptoms"]}
                     Disease : {state["disease_predictor"]}
""")
    ]

    result = model.invoke(message)
    return {"specialist_recommender":result.content}

def explanation_generator(state: MedicalState):

    disease = state.get("disease_predictor", "Not Applicable (Emergency Case)")

    specialist = state.get(
        "specialist_recommender",
        "Immediate Emergency Care Required"
    )

    hospitals = state.get("hospitals", "No hospitals found")

    message = [
        HumanMessage(
            content=f"""
Generate a final report.

User Query:
{state['collect_input']}

Symptoms:
{state['extract_symptoms']}

Emergency:
{state['emergency']}

Disease:
{disease}

Recommended Specialist:
{specialist}

Nearest Hospitals:
{hospitals}

Generate a clear medical report.
"""
        )
    ]

    result = model.invoke(message)

    return {"final_report": result.content}

def route_emergency(state: MedicalState):

    if state["emergency"]:
        return "emergency_path"

    return "normal_path"



from langgraph.graph import StateGraph, START, END

builder = StateGraph(MedicalState)

# Nodes
builder.add_node("extract_symptoms", extract_symptoms)
builder.add_node("emergency", emergency)
builder.add_node("disease", disease)
builder.add_node("specialist", specialist)
builder.add_node("get_location", get_location)
builder.add_node("hospital_finder", hospital_finder)
builder.add_node("explanation_generator", explanation_generator)

# Initial Flow
builder.add_edge(START, "extract_symptoms")
builder.add_edge("extract_symptoms", "emergency")

# Conditional Branch
builder.add_conditional_edges(
    "emergency",
    route_emergency,
    {
        "emergency_path": "get_location",
        "normal_path": "disease",
    }
)

# Non-emergency path
builder.add_edge("disease", "specialist")
builder.add_edge("specialist", "get_location")

# Common path
builder.add_edge("get_location", "hospital_finder")
builder.add_edge("hospital_finder", "explanation_generator")
builder.add_edge("explanation_generator", END)

# Compile
graph = builder.compile()