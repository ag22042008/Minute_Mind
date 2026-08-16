# import whisper
# import os
# WHISPER_MODEL=os.getenv("WHISPER_MODEL","medium")
# # if we are running second time we have to prevent reload of the model
# _model=None
# SARVAM_API_KEY=os.getenv("SARVAM_API_KEY")
# def load_model():
#     global _model
#     if _model is None:
#         print("loading model...")
#         _model=whisper.load_model(WHISPER_MODEL)
#         print("WHISPER MODEL LOADED SUCCESSFULLY")
#     return _model
# def transcribe_chunk(chunk_path: str,translate:bool=False)->str: # returns transcription text
#     model=load_model()
#     task="translate"if translate else "transcribe" # ternary if else
#     # Transcribe this audio chunk using the Whisper model
# # chunk_path: path to the audio file to transcribe
# # task: "transcribe" (same language) or "translate" (to English)
#     result=model.transcribe(chunk_path,task=task) 
#     return result['text']

# # if there is a error in any chunk all fn will stop and error will known to us and if we wrap all in a single it will make us we will not be able to know error

# def transcribe_all(chunks:list,translate:bool=False)->str:
#     full_transcript=""
#     for i,chunk in enumerate(chunks):# i->index chunk->values
#         print(f"Transcribing chunk {i+1}")
#         text=transcribe_chunk(chunk,translate=translate)

#         full_transcript+=text+" "
#     print("Transcription completed")
#     return full_transcript
import os
import time
import logging
from dotenv import load_dotenv
from groq import Groq

# Ensure .env is prioritized over system variables.
load_dotenv(override=True)

def get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        import streamlit as st
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not key:
        raise ValueError("Cannot transcribe: GROQ_API_KEY is missing from both environment variables and Streamlit secrets.")
    return Groq(api_key=key)

def transcribe_chunk(chunk_path: str, translate: bool = False, max_workers: int = None) -> str:
    """
    Uploads an audio file/chunk to Groq API and gets the transcription directly.
    """
    client = get_groq_client()
        
    
    
    with open(chunk_path, "rb") as file:
        file_tuple = (os.path.basename(chunk_path), file.read())
        
        if translate:
            
            response = client.audio.translations.create(
                file=file_tuple,
                model="whisper-large-v3",
                response_format="text"
            )
        else:
           
            response = client.audio.transcriptions.create(
                file=file_tuple,
                model="whisper-large-v3",
                response_format="text"
            )
    
    # response is directly the string text when response_format="text"
    if isinstance(response, str):
        return response.strip()
    return response.text.strip()
    
def transcribe_all(chunks: list, translate: bool = False) -> str:
    """
    Processes all chunks sequentially.
    """
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        
        text = transcribe_chunk(chunk, translate=translate)
        full_transcript += text + " "
        
    
    return full_transcript

#  
