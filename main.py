from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from pprint import pprint
from dotenv import load_dotenv


class PersonalInfo(BaseModel):
    """Schema for storing personal information of the user."""

    name: str = Field(description="Full name of the user")
    age: int = Field(description="Age of the user")
    profession: str = Field(description="Profession of the user")


load_dotenv()

model = init_chat_model(model="gpt-4.1-mini", temperature=0.7)
agent = create_agent(model=model, response_format=PersonalInfo)


response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Extraia os dados: Cleyson Lima, 35 anos, engenheiro de software.",
            )
        ]
    }
)

pprint(response)

personal_info = response["structured_response"]

print(personal_info.name)
print(personal_info.age)
print(personal_info.profession)
