from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf(data):
    loader = DirectoryLoader(data,glob = "*.pdf",loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

def convert_data():
    extracted_documents = load_pdf("data/")

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 20)
    text_chunks = splitter.split_documents(extracted_documents)
    return text_chunks

if __name__ == "__main__":
    text_chunks = convert_data()
    print(len(text_chunks),"\n")
    print(text_chunks[:5])


    

