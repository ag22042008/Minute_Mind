# uv pip install -U "yt-dlp[default]" --link-mode=copy 
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import key_decisions,extract_questions,actionable_items
from core.summarise import summarize
source=r"https://youtu.be/ERbKUVbhnf8?si=pb-4XDmigQoUKaDM"
chunks=process_input(source)
full_transcript=transcribe_all(chunks)

print(summarize(full_transcript))
print("="*20+"actionable_items"+"="*20)
print(actionable_items(full_transcript))
print("="*20+"extract_questions"+"="*20)

print(extract_questions(full_transcript))

print("="*20+"Key_decisions"+"="*20)
print(key_decisions(full_transcript))



