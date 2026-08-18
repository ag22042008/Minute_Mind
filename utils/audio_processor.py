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
# filename = ydl.prepare_filename(info)
# base, _ = os.path.splitext(filename)
# filename = base + ".wav"

# This just swaps whatever the extension is for .wav, instead of guessing between two specific ones — safer regardless of what format YouTube happens to serve.

# 6. Returning the path

# python
# return filename

# Finally, it hands back the full path to the .wav file so the caller can use it (e.g., pass it to a transcription tool).

# .set_channels(1) → converts the audio to mono (1 channel), regardless of whether it was originally stereo (2 channels) or something else. If it was stereo, this merges both channels into one.
# .set_frame_rate(16000) → resamples the audio so it has a sample rate of 16,000 Hz (16kHz), meaning 16,000 audio samples per second, regardless of what the original rate was (commonly 44100 Hz or 48000 Hz for normal audio) for scaling it acc to whisper scale.


import yt_dlp
# pyrefly: ignore [missing-import]
from pydub import AudioSegment
import os
import tempfile
import streamlit as st

# Explicitly set ffmpeg/ffprobe paths so pydub finds them on Streamlit Cloud,
# where PATH may not include /usr/bin by default.
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def get_cookies_path():
    """Safely fetch cookies from Streamlit Secrets into a temporary file."""
    try:
        cookies_content = st.secrets.get("www.youtube.com_cookies.txt", None)
        if cookies_content:
            print(f"✅ Found YouTube cookies in Streamlit Secrets! Length: {len(cookies_content)} chars")
            fd, path = tempfile.mkstemp(suffix=".txt", text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(cookies_content)
            print(f"✅ Temporary cookie file created at: {path}")
            return path
        else:
            print("⚠️ No cookies found in Streamlit Secrets under key 'www.youtube.com_cookies.txt'")
    except Exception as e:
        print(f"⚠️ Error reading Streamlit Secrets: {e}")
    return None


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        # Pull best available audio stream, fall back to best overall
        "format": "bestaudio/best",

        # Output file naming template — uses video title
        "outtmpl": output_path,

        # Bypass 403 Forbidden errors triggered by Cloud IPs (Streamlit).
        # We prioritize iOS/Safari which historically don't require the strict Node.js PO token challenge.
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "web_safari", "android"],
            }
        },

        # After downloading, run FFmpeg to convert to WAV
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        # Force 16kHz mono at the FFmpeg conversion stage (Whisper-optimal)
        "postprocessor_args": {
            "extractaudio": ["-ar", "16000", "-ac", "1"]
        },

        "quiet": True,
    }

    # Inject the temporary cookies file into yt-dlp if it exists in Streamlit secrets
    cookies_file = get_cookies_path()
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
        # mweb (mobile web) and web_creator use different URL signing than the main
        # web client, and may handle n-challenge differently. Both support cookies.
        ydl_opts["extractor_args"]["youtube"]["player_client"] = ["mweb", "web_creator"]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Swap the original container extension for .wav
        base, _ = os.path.splitext(filename)
        filename = base + ".wav"

    # For security, delete the temporary cookies file off the server disk immediately
    if cookies_file and os.path.exists(cookies_file):
        os.remove(cookies_file)

    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any video/audio file to 16kHz mono WAV using ffmpeg subprocess.
    Streams through disk — never loads the full file into Python memory.
    Far more RAM-efficient than pydub for large files."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    import subprocess
    result = subprocess.run(
        [
            "/usr/bin/ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",   # 16 kHz sample rate (Whisper-optimal)
            "-ac", "1",       # mono
            "-vn",            # no video stream
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 12) -> list:
    """Split a WAV file into chunks using ffmpeg segment muxer.
    Writes each chunk directly to disk — never loads the whole file into RAM."""
    import subprocess
    chunk_ms = chunk_minutes * 60  # ffmpeg segment_time is in seconds
    base = os.path.splitext(wav_path)[0]
    pattern = f"{base}_chunk_%03d.wav"

    result = subprocess.run(
        [
            "/usr/bin/ffmpeg", "-y",
            "-i", wav_path,
            "-f", "segment",
            "-segment_time", str(chunk_ms),
            "-c", "copy",
            pattern,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg chunking failed: {result.stderr[-500:]}")

    # Collect all written chunk files in order
    chunks = sorted(
        f for f in (
            os.path.join(os.path.dirname(wav_path), fn)
            for fn in os.listdir(os.path.dirname(wav_path) or ".")
        )
        if os.path.basename(f).startswith(os.path.basename(base) + "_chunk_")
        and f.endswith(".wav")
    )
    return chunks


def process_input(source: str) -> list:
    # Clean accidental brackets, quotes, or spaces from copy-pasting
    source = source.strip().strip('[]"\', ')

    # Trigger function to activate all functions in one go
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to wav...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
