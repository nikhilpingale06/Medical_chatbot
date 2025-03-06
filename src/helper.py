from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#Extracting data from pdf
def load_pdf_data(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

#Splitting the extracted data into the chunks.
def split_data(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    chunks = splitter.split_documents(extracted_data)
    return chunks

def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings