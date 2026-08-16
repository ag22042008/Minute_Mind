import os 
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
model="mistral-embed"
# directory where we save everything
CHROMA_DIR="vector_db"
# where we will save all the transcripts
COLLECTION_NAME="meeting_transcript"
# creating the vector store
def embedding_model():
    return MistralAIEmbeddings(model=model)
def build_vector_store(transcript:str)->Chroma:
    print("Building Vector Store")
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks=splitter.split_text(transcript)
    docs=[
       Document(page_content=chunk,metadata={'chunkindex':i}) for i ,chunk in enumerate( chunks)
    ]
    embeddings=embedding_model()
    vector_store=Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vector_store

def load_vector_store()->Chroma:
    embeddings=embedding_model()
    vector_store=Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vector_store

#top k elements as retrived
def get_retriever(vector_store:Chroma,k: int =4):
    return vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={"k":k,"fetch_k":15,"lambda_mult":0.25},
    )
