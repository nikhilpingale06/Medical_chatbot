from src.helper import load_pdf_data, split_data, download_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_data(data ="data/")
chunks = split_data(extracted_data)
embeddings = download_embeddings()

#Initiallizing pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

#creating embeddings for chunks and storing it in pinecone db
from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name = index_name,
    embedding= embeddings   
)