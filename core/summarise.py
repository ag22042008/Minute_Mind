# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
import os
import time

def getllm_mistral():
    return ChatMistralAI(model="mistral-large-latest", temperature=0.2).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

def getllm_groq():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

def getllm_gemini():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )

def get_resilient_llm():
    # Primary: Gemini (higher limits), Fallbacks: Mistral, then Groq
    return getllm_gemini().with_fallbacks([getllm_mistral(), getllm_groq()])


def split_transcript(transcript:str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=15000,
        chunk_overlap=1000,
    )
    return splitter.split_text(transcript)

def summarize(transcript:str)->str:
    llm=get_resilient_llm()
    map_prompt=ChatPromptTemplate.from_messages(
        [
            ("system","Summarize this portion of a meeting transcript concisely and only in english"),
            ("human","{text}"),
        ]
    )
    map_chain=map_prompt|llm|StrOutputParser()# summarize each chunk section of a video
    chunks=split_transcript(transcript) 
    chunk_summaries = []
    for chunk in chunks:
        result = map_chain.invoke({"text": chunk})
        chunk_summaries.append(result)
        time.sleep(4) # Pauses for 4 seconds to let the API cool down
    combined="\n\n".join(chunk_summaries)
    # summaries may overlapp so we will sumaarise whole finally
    combined_prompt=ChatPromptTemplate.from_messages([
        ("system","You are a expert summariser .combine these partial summaries into a complete professional meeting summary in bullet points ."),
        ("human","{text}"),
    ])
    combined_chain = combined_prompt | llm | StrOutputParser()

    # Recursive reduce: keep summarizing chunks until we have one final chunk that fits in the context window
    combined_chunks = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200).split_text(combined)
    
    while len(combined_chunks) > 1:
        reduced_summaries = []
        for chunk in combined_chunks:
            res = combined_chain.invoke({"text": chunk})
            reduced_summaries.append(res)
            time.sleep(4) # Pauses for 4 seconds
        new_combined = "\n\n".join(reduced_summaries)
        combined_chunks = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200).split_text(new_combined)
        
    return combined_chain.invoke({"text": combined_chunks[0]})

def generate_title(transcript:str)->str:
    llm=get_resilient_llm()
    title_prompt=ChatPromptTemplate.from_messages([
        ("system","Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else."),
        ("human","{text}")
    ])
    text=transcript[:2000]
    title_chain=title_prompt|llm|StrOutputParser()
    return title_chain.invoke({"text":text})