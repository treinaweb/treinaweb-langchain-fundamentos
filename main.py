from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


model = init_chat_model(model="gpt-4.1-mini", temperature=0.7)

messages = [
    SystemMessage(
        content="Você é um assistente que responde apenas em forma de haiku."
    ),
    HumanMessage(content="Como funciona o Langchain Framework?"),
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
