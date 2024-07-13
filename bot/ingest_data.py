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


if __name__ == "__main__":
    vector_store = ingest("done")#"done" HERE BECAUSE I HAD ALREADY CREATED DB IN experiments.py
    print("DB has been initialized")

