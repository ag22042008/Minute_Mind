#What each part does

# 1. The function signature

# python
# def download_youtube_audio(url:str)->str:

# It takes a YouTube URL (a string) as input, and promises to return a string (the path to the downloaded file) when it's done.

# 2. Building the output path template

# python
# output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

# This creates a "template" for where the file should be saved. DOWNLOAD_DIR is your download folder, and %(title)s.%(ext)s are placeholders that yt_dlp (the YouTube downloader library) will fill in later — %(title)s becomes the video's title, and %(ext)s becomes whatever file extension it ends up downloading (like .webm or .m4a).

# So if a video is called "Cool Song" and downloads as webm, the file initially becomes Cool Song.webm.

# 3. Setting the download options

# python
# ydl_opts = {
#     "format": "bestaudio/best",
#     "outtmpl": output_path,
#     "postprocessors": [...],
#     "quiet": True,
# }

# This is a settings dictionary telling yt_dlp how to behave:

# "format": "bestaudio/best" → grab the best available audio-only stream (falls back to best overall if no audio-only option exists).
# "outtmpl": output_path → use the naming template from step 2.
# "postprocessors" → after downloading, run FFmpeg to convert the audio into .wav format at 192 kbps quality.
# "quiet": True → don't print a bunch of progress logs to the console.

# 4. Actually downloading

# python
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     info = ydl.extract_info(url, download=True)

# This creates a downloader object using those settings, then tells it to fetch info about the video and download it (download=True). info ends up holding metadata about the video (title, duration, format used, etc).

# 5. Figuring out the final filename

# python
# filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")

# This is a bit of a workaround. ydl.prepare_filename(info) reconstructs what the original filename would have been (e.g., Cool Song.webm) — but since the postprocessor already converted that file to .wav on disk, the code manually swaps .webm or .m4a in the name for .wav so it matches the real file that now exists.

# 6. Returning the path

# python
# return filename

# Finally, it hands back the full path to the .wav file so the caller can use it (e.g., pass it to a transcription tool).

# Step-by-step walkthrough with an example

# Say you call:

# python
# download_youtube_audio("https://youtube.com/watch?v=abc123")
# yt_dlp visits that URL and inspects available formats.
# It picks the best pure-audio stream (say the video's native audio is in .m4a).
# It downloads that audio stream and saves it temporarily as DOWNLOAD_DIR/My Video.m4a.
# FFmpeg (via the postprocessor) converts that file into DOWNLOAD_DIR/My Video.wav, then deletes the original .m4a.
# The code computes what it thinks the filename should be (My Video.m4a), then string-replaces .m4a → .wav to get My Video.wav.
# It returns "DOWNLOAD_DIR/My Video.wav".
# A subtle risk worth knowing

# The .replace() trick is fragile — it assumes the original extension was always .webm or .m4a. If YouTube serves a different container (e.g. .opus, .mp4, .ogg), the replace calls silently do nothing, and the returned filename will be wrong (it'll still say .opus even though the real file on disk is .wav).

# A more robust approach is to ask the postprocessor for the actual final path, e.g.:

# python
# filename = ydl.prepare_filename(info)
# base, _ = os.path.splitext(filename)
# filename = base + ".wav"

# This just swaps whatever the extension is for .wav, instead of guessing between two specific ones — safer regardless of what format YouTube happens to serve.

#     This line does two conversions, chained together:
#This tells pydub what encoding/container to use when writing the file. Even though output_path already ends in .wav, pydub doesn't automatically infer format from the filename extension — you have to explicitly tell it. Internally, pydub hands this off to FFmpeg, which does the actual encoding into the WAV format.

#in audio.export()If you left this out or set it to something else (like format="mp3") while still naming the file .wav, you'd get a file named something.wav that's actually encoded as mp3 internally — a mismatch that could confuse other programs trying to read it. So the format argument and the file extension in output_path need to agree.
# .set_channels(1) → converts the audio to mono (1 channel), regardless of whether it was originally stereo (2 channels) or something else. If it was stereo, this merges both channels into one.
# .set_frame_rate(16000) → resamples the audio so it has a sample rate of 16,000 Hz (16kHz), meaning 16,000 audio samples per second, regardless of what the original rate was (commonly 44100 Hz or 48000 Hz for normal audio) for scaling it acc to whisper scale.



import yt_dlp
# pyrefly: ignore [missing-import]
from pydub import AudioSegment
import os

DOWNLOAD_DIR='downloades'
output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
# COOKIES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cookies.txt"))
os.makedirs(DOWNLOAD_DIR,exist_ok=True)
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "extractor_args": {"youtube": {"player_client": ["default", "-tv", "web_safari", "web_embedded"]}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": {
            "extractaudio": ["-ar", "16000", "-ac", "1"]
        },
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        filename = base + ".wav"
    return filename


def convert_to_wav(input_path:str)->str:
    "Convert any other video type mp4 mp3 format to wav format using pydub.and also youtube audio to a refined audio and into a whisper readable format"
    output_path=os.path.splitext(input_path)[0]+"_converted.wav"
    audio=AudioSegment.from_file(input_path)# getting the audio form the video part
    audio=audio.set_channels(1).set_frame_rate(16000)#16khz
    audio.export(output_path,format="wav")
    return output_path

#chunking the large videos and as whisper cant process large files and chunking is done in milliseconds
def chunk_audio(wav_path:str,chunk_minutes: int =12)->list:
    audio=AudioSegment.from_wav(wav_path)#extracting the audio from file
    chunk_ms=chunk_minutes*60*1000
    chunks=[]

    for i ,start in enumerate(range(0,len(audio),chunk_ms)):
        chunk=audio[start:start+chunk_ms]
        chunk_path=f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path,format="wav")
        chunks.append(chunk_path)

    return chunks
def process_input(source:str)->list: 
    # Clean accidental brackets, quotes, or spaces from copy-pasting
    source = source.strip().strip('[]"\', ')
    
    #trigger function to activate all functions in one go
    if source.startswith("http://")or source.startswith("https://"):
        print("detected Youtube URL.Downloading audio...")
        wav_path=download_youtube_audio(source)
    else:
        print("Detected Local file.Converting to wav...")
        wav_path=convert_to_wav(source)

    print("Chunking audio.....")
    chunks=chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks








