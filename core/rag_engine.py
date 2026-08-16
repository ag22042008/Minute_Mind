#rag function based explanation
# Let's break this down piece by piece, very slowly.

# First, the pieces you're looking at
# python
# {"context" : retriever | RunnableLambda(format_docs), ...}

# This is a Python dictionary with a key "context". The value for that key is this expression:

# python
# retriever | RunnableLambda(format_docs)

# So the whole thing means: "the value stored under the key 'context' is not a plain piece of data — it's a mini pipeline that will get run later."

# Breaking down the value: retriever | RunnableLambda(format_docs)

# Think of | as an arrow: "then send the output to..."

# retriever  →  RunnableLambda(format_docs)
# retriever — this is your search tool. When it eventually receives a question (a string), it searches the vector store and outputs a list of Document objects (the top 4 matching transcript chunks).
# RunnableLambda(format_docs) — this is a wrapper around your plain function format_docs. All RunnableLambda does is say: "treat this ordinary Python function as a step in the pipeline, so it knows how to receive input from | and pass output onward." Without this wrapper, LangChain wouldn't know how to plug a regular function into the | chain.
# The | between them means: whatever retriever outputs (the list of Documents) becomes the input to format_docs.

# So this one line, retriever | RunnableLambda(format_docs), is really just a compressed way of writing:

# python
# def get_context(question):
#     docs = retriever.invoke(question)   # step 1: search
#     return format_docs(docs)             # step 2: join into one string
# Now, why is this inside a dict under "context"?

# Zoom out to the full picture:

# python
# {
#     "context": retriever | RunnableLambda(format_docs),
#     "question": RunnablePassthrough()
# }

# Both "context" and "question" receive the same input — the original question string the user asked. LangChain runs both branches on that same input, at the same time, and collects the results into a new dict:

# python
# {
#     "context": "<joined transcript chunks>",   # from the retriever+format branch
#     "question": "<the original question, unchanged>"  # passthrough just returns it as-is
# }

# That resulting dict is exactly what your prompt template needs next, since your prompt has two blanks to fill: {context} and {question}.

# One-sentence summary

# {"context" : retriever | RunnableLambda(format_docs), ...} means: "take the incoming question, search the transcript for relevant chunks, join those chunks into one text block, and store that text block under the key 'context' — so it's ready to be inserted into the prompt."

# what does this dict works i couldnt understand anything

# Thought for 5s

# Let's start from zero, no LangChain at all.

# A normal dictionary you already know
# python
# person = {"name": "Alice", "age": 30}

# Here, the values ("Alice", 30) are just plain data, sitting there. Nothing runs. It's static.

# The dict in your RAG chain is NOT like that
# python
# {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}

# Here, the values are not data — they're more like two little machines waiting to be turned on. Nothing has run yet. Think of it like this:

# python
# {"context": <machine #1>, "question": <machine #2>}
# What happens when you call .invoke("What did Alice say about the budget?")

# LangChain takes your input string and does this:

# Feeds the same input into every machine in the dict, one by one (conceptually at the same time):
# Machine #1 (retriever | RunnableLambda(format_docs)) receives: "What did Alice say about the budget?"
# It searches the transcript, finds chunks, joins them into text.
# It produces: "Alice: cut budget by 10%.\n\nBob: disagree..."
# Machine #2 (RunnablePassthrough()) receives: "What did Alice say about the budget?"
# It does nothing to it. It just hands it straight back, unchanged.
# It produces: "What did Alice say about the budget?"
# Then it builds a brand new dict, using each machine's output as the value:
# python
# {
#     "context": "Alice: cut budget by 10%.\n\nBob: disagree...",
#     "question": "What did Alice say about the budget?"
# }

# That's it. That's the whole trick. The dict syntax is just LangChain's shorthand for: "run these machines on the same input, and collect their outputs into a dict with these key names."

# Why does it need to build that dict at all?

# Because the very next step in your chain is:

# python
# | prompt

# And your prompt template looks like this:

# python
# """...
# Context from meeting transcript:
# {context}"""
# python
# ("human", "{question}")

# It has two blanks to fill: {context} and {question}. A prompt template fills its blanks from a dict — it looks for a key matching each blank's name. That's the only reason the dict needs to have exactly the keys "context" and "question" — those names must match the {context} and {question} placeholders in the prompt.

# If you wrote it with zero LangChain magic, it would just be:
# python
# def run(question):
#     context_value = format_docs(retriever.invoke(question))  # machine 1
#     question_value = question                                 # machine 2 (does nothing)

#     filled_dict = {"context": context_value, "question": question_value}
#     return filled_dict
#RunnableLambda(format_docs) was never doing anything special to format_docs itself — it was just a label saying "hey LangChain, this is a plain function, please treat it as a pipeline step."
#That filled_dict is exactly what gets handed to prompt next. The confusing part isn't really "how dicts work" — it's that LangChain lets you put unrun pipelines as dictionary values, and only runs them the moment .invoke() is called.



import os 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

from core.vectore_store import load_vector_store,build_vector_store,get_retriever,embedding_model

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.25,
        max_tokens=1024
    )

# format documents
def format_docs(docs):
    return"\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str):

  vector_store=build_vector_store(transcript)
  retriever=get_retriever(vector_store,6)
  llm=get_llm()
  prompt=ChatPromptTemplate.from_messages([
  ("system","""You are an expert meeting assistant. Answer the user's question 
  based ONLY on the meeting transcript context provided below.

   If the answer is not found in the context, say: 
  "I could not find this information in the meeting transcript."

  Always be concise and precise. If quoting someone, mention it clearly.

  Context from meeting transcript:
  {context}""")
  ,("human", "{question}"),
    ])
# full LCEL RAG pipeline
  rag_chain=(
    {
        "context": retriever|RunnableLambda(format_docs),
        "question":RunnablePassthrough()

    }|prompt|llm|StrOutputParser()
    )
  return rag_chain
# builded the chain as feeded the store with chunks now loading it to feed for different question
#build_rag_chain and load_rag_chain serve two different moments in your app's lifecycle, not the same purpose twice:
# build_rag_chain(transcript) — runs when you have a brand-new transcript for the first time. It calls build_vector_store(transcript), which presumably chunks the text and embeds it into a fresh vector store. This is the expensive "ingestion" step.
# load_rag_chain() — runs on subsequent questions/sessions about a transcript you've already processed. Instead of re-chunking and re-embedding the same transcript again (wasteful, costs API calls, slow), it calls load_vector_store() to pull the previously-built index back from disk (e.g. a persisted FAISS/Chroma store).

def load_rag_chain(transcript:str):
   vector_store=load_vector_store()
   retriver=get_retriever()
   llm=get_llm()
   prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])
   rag_chain=(
       {
           "context": retriver|RunnableLambda(format_docs()),
           "question":RunnablePassthrough()
   
       }|prompt|llm|StrOutputParser()
       )
   return rag_chain

# ask questions
def ask_questions(rag_chain,question:str)->str:
   
   answer=rag_chain.invoke(question)
  
   return answer


   








