#driver function file connecting all the modules
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarise import summarize,generate_title
from core.extractor import actionable_items,extract_questions,key_decisions
from core.rag_engine import build_rag_chain,ask_questions
load_dotenv()


def main_pipeline(url:str)->dict:
    print("*"*50+"Ai is starting to serve you"+"*"*50)
    chunks=process_input(url)
    transcription=transcribe_all(chunks)
    print(f"raw transcription (first 300 characters ) {transcription[:300]}")
    title=generate_title(transcription)
    Summary=summarize(transcription)
    actions_to_be_taken=actionable_items(transcription)
    questions_asked= extract_questions(transcription)
    decisions_imp=key_decisions(transcription)
    rag_chain = build_rag_chain(transcription)
    return {
        "title": title,
        "transcript": transcription,
        "summary": Summary,
        "action_items": actions_to_be_taken,
        "key_decisions": decisions_imp,
        "open_questions": questions_asked,
        "rag_chain": rag_chain,
    }

if __name__=="__main__":
    
    source = input("Enter YouTube URL or local file path: ").strip()
    result = main_pipeline(source)
    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)
    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'e' to quit)\n")
    rag=result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in "e":
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_questions(rag, question)
        print(f"\n🤖 Assistant: {answer}\n")





    





    











