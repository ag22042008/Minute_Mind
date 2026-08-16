#actionable items-> in a video if we have to perform an action In the context of this project, actionable items refer to specific tasks, follow-ups, or responsibilities mentioned during a meeting or video that require someone to take action
#In the context of this project, actionable items refer to specific tasks, follow-ups, or responsibilities mentioned during a meeting or video that require someone to take action
#decisions taken ,questions asked in meetings
from tokenize import String

from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import os
def getllm():
   return ChatMistralAI(model="mistral-large-latest",mistral_api_key=os.getenv("MISTRAL_API_KEY"),temperature=0.2)
def getllm2():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.2).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

# functions for different type for chains
def chain(system_prompt:str):
    llm=getllm2()
    return(ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{text}")
    ])|llm|StrOutputParser())

def splitter_text():
    # Increased chunk_size to drastically reduce the number of API calls, stopping the 45min bottleneck
    return RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)

# extract actionable_items
def actionable_items(transcript:str)->str:
    splitter=splitter_text()
    chunks=splitter.split_text(transcript)
    re_chain =chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    actionable_items=""
    for chunk in chunks:
        actionable_items+=re_chain.invoke({"text":chunk})
    return actionable_items



#extract_key_decisions
def key_decisions(transcript:str)->str:
    splitter=splitter_text()
    chunks=splitter.split_text(transcript)
    chain_key=chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    key_decisions=""
    for chunk in chunks:
        key_decisions+=chain_key.invoke({"text":chunk})
    return key_decisions

def extract_questions(transcript: str) -> str:
     splitter=splitter_text()
     chunks=splitter.split_text(transcript)
     chain_extract_unresolved_questions = chain("From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'")
     unresolved_questions=""
     for chunk in chunks:
         unresolved_questions+=chain_extract_unresolved_questions.invoke({"text":chunk})
     return unresolved_questions

     
    
    

