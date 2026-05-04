from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()


model = init_chat_model(model="gpt-4.1-mini", temperature=0.7)

response = model.invoke("O que é o Langchain Framework?")

print(response.content)
