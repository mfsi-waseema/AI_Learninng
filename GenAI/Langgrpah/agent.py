from dotenv import load_dotenv
from typing import Annotated

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------
# Load env
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# LLM
# ---------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

# ---------------------------------------------------
# Tools
# ---------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a math expression.
    Example: 2 + 2 * 10
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather(city: str) -> str:
    """
    Fake weather tool example.
    """
    return f"The weather in {city} is 30°C and sunny."


tools = [calculator, weather]

# Bind tools to model
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------
# Graph State
# ---------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------------------------------
# Chatbot Node
# ---------------------------------------------------

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }

# ---------------------------------------------------
# Conditional Routing
# ---------------------------------------------------

def route_tools(state: State):
    last_message = state["messages"][-1]

    # If model wants to call tools
    if last_message.tool_calls:
        return "tools"

    return END

# ---------------------------------------------------
# Build Graph
# ---------------------------------------------------

builder = StateGraph(State)

# Nodes
builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
builder.add_node("tools", tool_node)

# Entry
builder.set_entry_point("chatbot")

# Conditional edge
builder.add_conditional_edges(
    "chatbot",
    route_tools
)

# Return to chatbot after tool execution
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot",END)


# ---------------------------------------------------
# Memory
# ---------------------------------------------------

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)

# ---------------------------------------------------
# Chat Loop
# ---------------------------------------------------

config = {
    "configurable": {
        "thread_id": "user-123"
    }
}

print("\nLangGraph + Groq Agent Started")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config
    )

    print("\nAI:")
    print(result["messages"][-1].content)
    print()