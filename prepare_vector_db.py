from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

data_path = "data"
vector_db_path = "vector-store/db-faiss"

def create_db_from_files():
    # Define loader to scan all data directory
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documnents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
    chunks = text_splitter.split_documents(documnents)

    # Embedding chunks
    embedding_model = GPT4AllEmbeddings(
        model_file="models/all-MiniLM-L6-v2-f16.gguf",
        gpt4all_kwargs={'allow_download': 'True'},
        device="gpu"
    )
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_files()





