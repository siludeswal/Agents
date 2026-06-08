import langchain
import langgraph
from langgraph.graph import StateGraph,END,START
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from typing import TypedDict
class PortfolioState(TypedDict):
    resume_text:str
    resume_information: str
    personal_information:str
    skill:str
    score:float
    final_decision:str
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant")
def resume_information(state: PortfolioState):

    result = llm.invoke([
        HumanMessage(
            content=f"Get all information from:\n{state['resume_text']}"
        )
    ])

    return {
        "resume_information": result.content
    }

def personal_information(state:PortfolioState)-> PortfolioState:
    """ Extract all the personal information from the given data."""
    message = [HumanMessage(content=f" Extract the personal information from this {state["resume_text"]}")]
    result = llm.invoke(message)
    return {
        "personal_information": result.content
    }

def skills(state: PortfolioState):

    result = llm.invoke([
        HumanMessage(
            content=f"Extract all skills from:\n{state['resume_information']}"
        )
    ])

    return {
        "skill": result.content
    }

def score(state):

    result = llm.invoke([
        HumanMessage(
            content=f"""
            Skills:
            {state['skill']}

            Give ONLY a numeric score between 0 and 10.
            Return nothing except the number.

            Example:
            8.5
            """
        )
    ])

    return {
        "score": float(result.content.strip())
    }

def final_decision(state: PortfolioState):

    if state["score"] >= 7:
        decision = "Shortlisted"
    else:
        decision = "Rejected"

    return {
        "final_decision": decision
    }

tools = [resume_information,personal_information,skills,score,final_decision]
model = llm.bind_tools(tools=tools)
from langgraph.prebuilt import ToolNode, tools_condition
builder = StateGraph(PortfolioState)

builder.add_node("information",resume_information)
builder.add_node("personal information",personal_information)
builder.add_node("skills",skills)
builder.add_node("score",score)
builder.add_node("decision",final_decision)

builder.add_edge(START,"information")
builder.add_edge("information","personal information")
builder.add_edge("personal information","skills")
builder.add_edge("skills","score")
builder.add_edge("score","decision")
builder.add_edge("decision",END)

graph = builder.compile()

