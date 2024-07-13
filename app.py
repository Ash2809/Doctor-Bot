from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import sys
# from bot.retrieval_gen import generation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from bot.data_converter import convert_data
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os

def ingest(status):
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ASTRA_DB_API = os.getenv("ASTRA_TOKEN")
    ASTRA_ENDPOINT = os.getenv("DB_ENDPOINT")

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    vector_store = AstraDBVectorStore(token = ASTRA_DB_API,
                                      api_endpoint = ASTRA_ENDPOINT,
                                      embedding = gemini_embeddings,
                                      namespace = "default_keyspace",
                                      collection_name = "Medical")
    is_full = status
    if is_full == None:#THIS MEANS THERE IS NO VECTORS CREATED IN DB
        text_chunks = convert_data()
        inserted_ids = vector_store.add_documents(text_chunks)
    else:
        return vector_store
    
    
    return vector_store,inserted_ids

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

app = Flask(__name__)

load_dotenv()

vector_store=ingest("done")
chain=generation(vector_store)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result=chain.invoke(input)
    print("Response : ", result)
    return str(result)

if __name__ == '__main__':
    print("Current Working Directory: ", os.getcwd())

    app.run(debug= True)