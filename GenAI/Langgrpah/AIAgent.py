from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq

from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Groq model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state):
        return {
            "messages": [model.invoke(state["messages"])]
        }

    graph_workflow.add_node("agent", call_model)

    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()

    return agent


def make_alternative_graph():
    """Tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])

    model_with_tools = model.bind_tools([add])

    def call_model(state):
        return {
            "messages": [
                model_with_tools.invoke(state["messages"])
            ]
        }

    def should_continue(state: State):
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)

    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("tools", "agent")

    graph_workflow.add_conditional_edges(
        "agent",
        should_continue
    )

    agent = graph_workflow.compile()

    return agent

agent = make_alternative_graph()