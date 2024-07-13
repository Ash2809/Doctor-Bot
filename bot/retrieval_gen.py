from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from ingest_data import ingest
from dotenv import load_dotenv
import os


def generation(vector_store):
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(temperature = 0.4,model = "gemini-pro")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    template = """You are a medical Chatbot who is expert in medicine knowledge and cure Context: {context}
    Question: {question}.Answer according to context.
    """

    prompt=ChatPromptTemplate.from_template(template)
    chain = ({"context":retriever,"question":RunnablePassthrough()}|prompt|llm|StrOutputParser())

    return chain

if __name__ == "__main__":
    vector_store = ingest("done")
    chain = generation(vector_store)

    response = chain.invoke("What is Malaria")
    print(response)
