from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000))

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

agent = create_react_agent(llm, tools=[search_tool, wiki_tool])

# When is the IPL Final?
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

response = agent.invoke({"messages": [("user", "Who is Kalpana Chawla and what is she known for?")]})
print(response["messages"][-1].content)
