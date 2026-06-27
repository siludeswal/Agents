from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()
import os
from IPython.display import Image, display, Markdown
from langchain_groq import ChatGroq
from datetime import datetime
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ── LLM & Tools ──────────────────────────────────────────────────────────────

llm = ChatGroq(model="llama-3.1-8b-instant")
tavily = TavilySearch(max_results=5)

@tool
def tavily_fact(query: str) -> str:
    """Search for latest news and factual information about the query."""
    result = tavily.invoke({"query": query})
    return result

llm_with_tool = llm.bind_tools([tavily_fact])

# ── State ─────────────────────────────────────────────────────────────────────
# FIX: was `messages = Annotated[...]` (assignment), must be `messages: Annotated[...]` (annotation)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    category: str
    final_result: str

# ── Nodes ─────────────────────────────────────────────────────────────────────

def category(state: AgentState):
    messages = [
        SystemMessage(content="""
You are a routing assistant.

Classify the user's request into ONLY one of these categories:
  general  →  greetings, coding, AI questions, explanations, casual chat
  news     →  latest/today's/yesterday's news, current affairs, recent events
  fact     →  verify a statement, check if something is true, detect misinformation

Return ONLY the single lowercase word — nothing else.
"""),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {"category": response.content.strip().lower()}


def general_chat(state: AgentState):
    messages = [
        SystemMessage(content=f"""
You are a helpful AI assistant.
Today's date is {datetime.now().strftime('%Y-%m-%d %H:%M')}.
Answer naturally using your own knowledge.
"""),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    print("General Node")
    return {
        "messages": [AIMessage(content=response.content)],
        "final_result": response.content,
    }


def latest_news(state: AgentState):
    query = state["messages"][-1].content
    try:
        search_result = tavily_fact.invoke(query)
    except Exception:
        search_result = "Unable to retrieve latest news."

    messages = [
        SystemMessage(content=f"""
You are a professional News Assistant.
Today's date is {datetime.now().strftime('%Y-%m-%d %H:%M')}.
Use the provided search results to answer the user's request.
Always summarize the news clearly.
"""),
        *state["messages"],
        HumanMessage(content=f"""
Latest Search Results:
{search_result}

Summarize the latest news. Include:
- Main Headlines
- Important Details
- Short Summary
"""),
    ]
    response = llm.invoke(messages)
    print("News Node")
    return {
        "messages": [AIMessage(content=response.content)],
        "final_result": response.content,
    }


def facts(state: AgentState):
    query = state["messages"][-1].content
    try:
        search_result = tavily_fact.invoke(query)
    except Exception:
        search_result = "Unable to retrieve supporting evidence."

    messages = [
        SystemMessage(content=f"""
You are an AI Fact Checker.
Today's date is {datetime.now().strftime('%Y-%m-%d %H:%M')}.
Use the search evidence to verify the user's statement.
"""),
        *state["messages"],
        HumanMessage(content=f"""
Search Evidence:
{search_result}

Verify the user's latest statement and respond in this format:

Verdict: (True / False / Misleading / Partially True)

Reason:

Evidence:
"""),
    ]
    response = llm.invoke(messages)
    print("Fact Node")
    return {
        "messages": [AIMessage(content=response.content)],
        "final_result": response.content,
    }

# ── Router ────────────────────────────────────────────────────────────────────

def router(state: AgentState) -> str:
    cat = state["category"]
    if "general" in cat:
        return "general_talk"
    elif "news" in cat:
        return "news"
    else:
        return "facts"

# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("category",    category)
builder.add_node("general_chat", general_chat)
builder.add_node("latest_news",  latest_news)
builder.add_node("fact",         facts)

builder.add_edge(START, "category")
builder.add_conditional_edges(
    "category",
    router,
    {
        "general_talk": "general_chat",
        "news":         "latest_news",
        "facts":        "fact",
    },
)
builder.add_edge("general_chat", END)
builder.add_edge("latest_news",  END)
builder.add_edge("fact",         END)

# FIX: pass checkpointer here to enable persistent memory across turns
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# ── Helper ────────────────────────────────────────────────────────────────────

def agent_response(query: str, thread_id: str = "default") -> str:
    """
    Invoke the graph with memory. Each unique thread_id maintains its own
    conversation history, so you can run multiple independent sessions.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    return result["final_result"]

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # All three calls share thread_id "session1" → full memory across turns
    print(agent_response("Hello! How are you?",              thread_id="session1"))
    print(agent_response("What is the latest AI news?",      thread_id="session1"))
    print(agent_response("The Sun revolves around the Earth?", thread_id="session1"))