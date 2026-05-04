from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv

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
models_with_tools = model.bind_tools([calculate_bmi])

messages = [
    HumanMessage(content="Qual o IMC de alguém com 70Kg e 1.75m?"),
]

response = models_with_tools.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    result = calculate_bmi.invoke(tool_call)
    messages.append(result)

final = models_with_tools.invoke(messages)
print(final)
