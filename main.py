from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

loader = CSVLoader(file_path="products.csv", encoding="utf-8")
documents = loader.load()

print(f"Loaded Documents: {len(documents)}")
print(f"First Document: {documents[0].page_content}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_products(query: str) -> str:
    """Searches the vector store for products related to the given query.

    Args:
        query: User search text used to retrieve relevant product documents.

    Returns:
        A formatted string containing up to 3 matching product entries,
        separated by dividers. Returns a fallback message when no matches
        are found.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant products found."

    result = [doc.page_content for doc in docs]
    return "\n\n---\n\n".join(result)


system_prompt = (
    "Você é um assistente de vendas especializado. "
    "Use a ferramenta search_products para consultar o catálogo antes de "
    "responder perguntas sobre os produtos. Sempre mencione o preço quando "
    "disponível."
)
agent = create_agent(
    model="gpt-4.1-mini",
    tools=[search_products],
    system_prompt=system_prompt,
)

questions = [
    "Quais produtos você tem para melhorar a ergonomia do home office?",
    "Estou montando um setup gamer, tem algo para recomendar?",
    "Preciso de periféricos sem fio, quais opções vocês têm?",
    "Qual o produto mais barato do catálogo?",
]

for question in questions:
    print(f"\n{'=' * 60}")
    print(f"Pergunta: {question}")

    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Resposta: {response['messages'][-1].content}")
