from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gpt-4.1-mini", temperature=0.7)
agent = create_agent(model=model, checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "user-123"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="Olá, meu nome é Cleyson")]}, config=config
)
print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [HumanMessage(content="Qual é o meu nome?")]}, config=config
)
print(response["messages"][-1].content)
