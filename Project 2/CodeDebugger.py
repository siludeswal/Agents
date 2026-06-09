import langchain
import langgraph
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import tool_node , tools_condition
from typing import TypedDict
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
from IPython.display import display , Image
llm = ChatGroq(model="llama-3.1-8b-instant")
class PortFolioState(TypedDict):
    input_code:str
    error:str
    improvement:str
    final_observations:str

def input(state:PortFolioState)-> PortFolioState:
    messsage = [SystemMessage(content="You are an AI Assitance to debug Code provided by the user."),
                HumanMessage(content=f"""Debug the code which is provided to you By the input {state['input_code']}
                            After debugging the code only provide the errors in the code.""")]
    response = llm.invoke(messsage)
    return {"error":response.content}

def improvement(state:PortFolioState)-> PortFolioState:
    message = [HumanMessage(content=F"""
                            Read the complete code from the {state['input_code']} and 
                            all the errors from {state['error']} and then List all the Improvement Required in the code.
                                """)]
    response = llm.invoke(message)
    return {"improvement":response.content}

def summary(state:PortFolioState)-> PortFolioState:
    message = [HumanMessage(content=f""" Read all the data from {state['input_code']}, {state['error']} and {state['improvement']}
                        After all this create a final code with all the updated codes, No error and tell me what are the major improvement you did.
                            """)]
    response = llm.invoke(message)
    return {"final_observations":response.content}

builder = StateGraph(PortFolioState)
builder.add_node("input_code",input)
builder.add_node("code improvement",improvement)
builder.add_node("summary",summary)

builder.add_edge(START,'input_code')
builder.add_edge("input_code","code improvement")
builder.add_edge("code improvement","summary")
builder.add_edge("summary",END)

graph = builder.compile()

