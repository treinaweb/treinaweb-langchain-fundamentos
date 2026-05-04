from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain.tools import tool
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


@tool
def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculate Body Mass Index (BMI).

    Args:
        weight: Body weight in kilograms.
        height: Body height in meters.

    Returns:
        BMI value rounded to 2 decimal places.
    """
    bmi = weight / (height**2)
    return round(bmi, 2)


model = init_chat_model(model="gpt-4.1-mini", temperature=0.7)
agent = create_agent(
    model=model,
    tools=[calculate_bmi],
    system_prompt="Você é um assistente que ajuda a calcular o IMC (Índice de Massa Corporal) com base no peso e altura fornecidos. Use a ferramenta de cálculo de IMC quando necessário.",
)

messages = [
    HumanMessage(content="Qual o IMC de alguém com 70Kg e 1.75m?"),
]

# response = agent.invoke({"messages": messages})

# print(response["messages"][-1].content)

# pprint(response)

for chunk in agent.stream({"messages": messages}, stream_mode="values"):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"Human: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"AI: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
