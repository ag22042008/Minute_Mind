"""
MinuteMind
----------
Turn any meeting recording into instant summaries, action items, key
decisions, open questions, and a chatbot that remembers everything discussed.

Run with:
    streamlit run app.py

Expects to sit alongside your existing project structure:
    utils/audio_processor.py
    core/transcriber.py
    core/summarise.py
    core/extractor.py
    core/rag_engine.py
    .env
"""

import os
import re
import html
import random
import tempfile
import shutil

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarise import summarize, generate_title
from core.extractor import actionable_items, extract_questions, key_decisions
from core.rag_engine import build_rag_chain, ask_questions

load_dotenv()

# ==========================================================================
# Page config
# ==========================================================================
st.set_page_config(
    page_title="MinuteMind",
    page_icon="▮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Design tokens + component CSS
# ==========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg:        #0b0e14;
    --surface:   #12161f;
    --surface-2: #1a1f2b;
    --border:    #242a38;
    --accent:    #6c5ce7;   /* indigo   — primary channel color */
    --accent-2:  #ffb454;   /* amber    — meter peak / highlight */
    --live:      #ff5470;   /* recording red */
    --success:   #34d399;
    --text:      #e7e9f2;
    --text-muted:#808a9e;
}

html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; }
.stApp { background: var(--bg) !important; }

* { font-family: 'Inter', sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

h1, h2, h3, .display { font-family: 'Space Grotesk', sans-serif !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* focus visibility */
button:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--accent-2) !important;
    outline-offset: 2px !important;
}

/* ---------- brand / eyebrow ---------- */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--accent-2);
    margin-bottom: 0.4rem;
}
.brand-mark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

/* ---------- hero ---------- */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(2.1rem, 4.5vw, 3.4rem);
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin: 0;
    background: linear-gradient(120deg, #ffffff 0%, var(--accent-2) 55%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-tagline {
    color: var(--text-muted);
    font-size: 0.92rem;
    max-width: 460px;
    margin-top: 0.35rem;
    line-height: 1.6;
}

/* animated hero waveform — the signature element */
.hero-wave { display: flex; align-items: flex-end; gap: 3px; height: 46px; margin: 1.1rem 0 1.6rem; }
.hero-wave .hbar {
    width: 4px; border-radius: 2px; height: var(--h);
    background: linear-gradient(180deg, var(--accent-2), var(--accent));
    transform-origin: bottom;
    animation-name: wavepulse;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
}
@keyframes wavepulse { 0%, 100% { transform: scaleY(0.45); } 50% { transform: scaleY(1); } }
@media (prefers-reduced-motion: reduce) { .hero-wave .hbar { animation: none; } }

/* ---------- console / signal-chain (sidebar) ---------- */
.chain-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 0.15rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.76rem;
}
.chain-row:last-child { border-bottom: none; }
.chain-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    width: 1.4rem;
}
.chain-label { flex: 1; color: var(--text); }
.chain-label.pending { color: var(--text-muted); }
.mini-wave { display: flex; align-items: flex-end; gap: 2px; height: 14px; }
.mini-wave .mbar { width: 3px; border-radius: 1px; }
.mbar.done    { background: var(--accent-2); }
.mbar.pending { background: var(--border); }

/* ---------- channel cards ---------- */
.channel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.channel::before {
    content: '';
    position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.channel-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: var(--accent-2);
    margin-bottom: 0.6rem;
}
.channel-body {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--text);
}
.channel-body ul { margin: 0.3rem 0 0.3rem 1.1rem; padding: 0; }
.channel-body li { margin-bottom: 0.3rem; }
.channel-body strong { color: var(--accent-2); font-weight: 600; }
.channel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    color: var(--text);
}

/* ---------- badges (empty state) ---------- */
.badge {
    display: inline-block; padding: 0.28rem 0.7rem; border-radius: 5px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    border: 1px solid var(--border); color: var(--text-muted);
    margin-right: 0.4rem;
}

/* ---------- inputs / buttons ---------- */
.stTextInput > div > div > input, .stTextArea textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; }

.stButton > button {
    background: linear-gradient(120deg, var(--accent), #5142c4) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.3rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(108,92,231,0.35) !important; }
.stButton > button[kind="secondary"] { background: var(--surface-2) !important; border: 1px solid var(--border) !important; }

[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 8px !important; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.4rem 0 !important; }
label { color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* ---------- live chat channel ---------- */
.led {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--live); margin-right: 0.4rem;
    animation: ledpulse 1.4s ease-in-out infinite;
}
@keyframes ledpulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
@media (prefers-reduced-motion: reduce) { .led { animation: none; } }

.chat-log { max-height: 380px; overflow-y: auto; margin-bottom: 0.9rem; }
.msg { margin-bottom: 0.85rem; display: flex; flex-direction: column; }
.msg-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem;
}
.msg.user { align-items: flex-end; }
.msg.user .msg-label { color: var(--accent-2); }
.msg.assistant .msg-label { color: var(--accent); }
.bubble { padding: 0.6rem 0.95rem; border-radius: 9px; font-size: 0.87rem; line-height: 1.6; max-width: 88%; white-space: pre-wrap; }
.msg.user .bubble { background: rgba(255,180,84,0.1); border: 1px solid rgba(255,180,84,0.25); }
.msg.assistant .bubble { background: rgba(108,92,231,0.12); border: 1px solid rgba(108,92,231,0.28); }

.empty-panel {
    text-align: center; padding: 3.5rem 1.5rem;
    border: 1px dashed var(--border); border-radius: 10px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Small render helpers
# ==========================================================================
def hero_waveform(n: int = 26) -> str:
    rnd = random.Random(7)
    bars = []
    for _ in range(n):
        h = rnd.randint(25, 90)
        delay = round(rnd.uniform(0, 1.3), 2)
        dur = round(rnd.uniform(0.9, 1.6), 2)
        bars.append(
            f'<span class="hbar" style="--h:{h}%; animation-delay:{delay}s; animation-duration:{dur}s;"></span>'
        )
    return f'<div class="hero-wave">{"".join(bars)}</div>'


def mini_wave(done: bool) -> str:
    heights = [40, 70, 55, 85] if done else [20, 20, 20, 20]
    cls = "done" if done else "pending"
    bars = "".join(f'<span class="mbar {cls}" style="height:{h}%"></span>' for h in heights)
    return f'<div class="mini-wave">{bars}</div>'


def esc(text: str) -> str:
    return html.escape(str(text)).replace("\n", "<br>")


def sanitize_content(text: str) -> str:
    """Defensively clean up whatever the backend functions hand back.

    Some LLM prompts wrap their answer in a ```code fence```, or echo back
    raw HTML wrapper tags, or occasionally duplicate a fallback sentence.
    None of that is meant to be shown to the user, so strip it here rather
    than let it leak into the UI as literal text.
    """
    text = str(text).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)     # leading ```lang fence
    text = re.sub(r"\n?```\s*$", "", text)               # trailing ``` fence
    text = re.sub(r"</?div[^>]*>", "", text, flags=re.IGNORECASE)  # stray <div>/</div>
    text = text.strip()

    # collapse "X.X." style exact-duplicate output into a single "X."
    n = len(text)
    if n > 1 and n % 2 == 0 and text[:n // 2] == text[n // 2:]:
        text = text[:n // 2]

    return text.strip()


def render_body(text: str) -> str:
    """Sanitize, then render a modest markdown subset (bold, bullets, line
    breaks) as safe HTML. Everything else is escaped so stray characters
    from the model can't break the layout."""
    text = html.escape(sanitize_content(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    out, in_list = [], False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("* ") or line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:].strip()}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(line)
    if in_list:
        out.append("</ul>")

    return "<br>".join(out) if out else "—"


def channel_card(tag: str, title_html: str, body: str):
    st.markdown(
        f"""<div class="channel">
<div class="channel-tag">{tag}</div>
{title_html}
<div class="channel-body">{render_body(body)}</div>
</div>""",
        unsafe_allow_html=True,
    )


CHAIN_STEPS = [
    ("01", "Capture audio"),
    ("02", "Transcribe speech"),
    ("03", "Name the session"),
    ("04", "Summarize"),
    ("05", "Extract insights"),
    ("06", "Index for chat"),
]

# ==========================================================================
# Session state
# ==========================================================================
DEFAULTS = {
    "processed": False,
    "title": "",
    "transcript": "",
    "summary": "",
    "action_items": "",
    "key_decisions": "",
    "open_questions": "",
    "rag_chain": None,
    "chat_history": [],
    "is_running": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_state():
    if st.session_state.get("is_running", False):
        st.session_state.is_running = False
        return
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    if os.path.exists("downloades"):
        try:
            shutil.rmtree("downloades")
        except Exception:
            pass


def run_pipeline(source: str) -> bool:
    """Runs the full pipeline. Returns True on success, False on failure
    (in which case a simple error message has already been shown)."""
    st.session_state.is_running = True
    status = st.status("Running the signal chain…", expanded=True)
    try:
        status.write("01 · Capturing audio…")
        chunks = process_input(source)

        status.write("02 · Transcribing speech…")
        transcription = transcribe_all(chunks)

        status.write("03 · Naming the session…")
        title = generate_title(transcription)

        status.write("04 · Summarizing…")
        summary = summarize(transcription)

        status.write("05 · Extracting action items, decisions & questions…")
        actions = actionable_items(transcription)
        decisions = key_decisions(transcription)
        questions = extract_questions(transcription)

        status.write("06 · Indexing for chat…")
        rag_chain = build_rag_chain(transcription)

        status.update(label="Signal locked. Meeting decoded.", state="complete", expanded=False)

        st.session_state.update(
            {
                "processed": True,
                "title": title,
                "transcript": transcription,
                "summary": summary,
                "action_items": actions,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "chat_history": [],
            }
        )
        return True

    except Exception as e:
        status.update(label="Pipeline failed", state="error", expanded=False)
        st.session_state.processed = False
        err = str(e)
        # Give a clear, actionable message for YouTube IP-blocking errors
        if any(kw in err for kw in ["403", "Forbidden", "format is not available", "PO Token", "n challenge", "page needs to be reloaded"]):
            st.error(
                "⚠️ YouTube blocked this request from the cloud server IP.\n\n"
                "**Fix:** Download the video/audio locally first (use [yt-dlp](https://github.com/yt-dlp/yt-dlp) "
                "or any YouTube downloader), then switch to **Upload file** in the sidebar and upload the file directly."
            )
        else:
            st.error(f"Something went wrong while processing this source: {e}")
        return False
        
    finally:
        st.session_state.is_running = False


# ==========================================================================
# Sidebar — console
# ==========================================================================
with st.sidebar:
    st.markdown('<div class="eyebrow">Audio → Insight</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-mark">▮ MinuteMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Meeting intelligence console</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    input_mode = st.radio("Source", ["YouTube URL", "Upload file"], horizontal=True, label_visibility="collapsed")

    source_value = None
    source_id = None
    if input_mode == "YouTube URL":
        url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
        if url:
            source_value = url
            source_id = f"url:{url}"
    else:
        uploaded = st.file_uploader(
            "Upload", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"], label_visibility="collapsed"
        )
        if uploaded is not None:
            source_id = f"file:{uploaded.name}:{uploaded.size}"
            # a different file was picked than the one currently loaded —
            # clear old results immediately so they don't linger on screen
            if st.session_state.get("last_source_id") != source_id:
                reset_state()
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source_value = tmp_path

    run_clicked = st.button("▸ Analyze", use_container_width=True, disabled=not source_value)
    st.button("Reset console", use_container_width=True, type="secondary", on_click=reset_state)

    if run_clicked and source_value:
        reset_state()  # always start from a clean slate before loading the new source
        st.session_state.last_source_id = source_id
        run_pipeline(source_value)

    # Sidebar note about YouTube URL limitations on cloud deployments
    if input_mode == "YouTube URL":
        st.markdown("""
<div style="background:rgba(255,180,84,0.08);border:1px solid rgba(255,180,84,0.25);
            border-radius:8px;padding:0.8rem 1rem;margin-top:0.8rem;font-size:0.75rem;
            color:var(--text-muted);line-height:1.6;">
<strong style="color:var(--accent-2);">ℹ️ YouTube on Cloud</strong><br/>
YouTube blocks direct audio downloads from cloud server IPs. If a URL fails, download the audio locally using 
<a href="https://github.com/yt-dlp/yt-dlp" target="_blank" style="color:var(--accent);">yt-dlp</a> 
or any downloader, then use <strong>Upload file</strong> instead.
</div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Signal chain</div>', unsafe_allow_html=True)
    for num, label in CHAIN_STEPS:
        done = st.session_state.processed
        st.markdown(
            f"""<div class="chain-row">
<span class="chain-num">{num}</span>
<span class="chain-label {'' if done else 'pending'}">{label}</span>
{mini_wave(done)}
</div>""",
            unsafe_allow_html=True,
        )

# ==========================================================================
# Main — hero
# ==========================================================================
st.markdown('<div class="eyebrow">Signal from your meetings</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">MinuteMind</div>', unsafe_allow_html=True)
st.markdown(hero_waveform(), unsafe_allow_html=True)
st.markdown(
    '<div class="hero-tagline">Turns any meeting recording into instant summaries, '
    'action items, key decisions, and a chatbot that remembers everything discussed.</div>',
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================================
# Main — content
# ==========================================================================
if not st.session_state.processed:
    st.markdown(
        """<div class="empty-panel">
<div class="brand-mark" style="font-size:1.3rem;margin-bottom:0.4rem;">Nothing to play back yet</div>
<div style="color:var(--text-muted);font-size:0.85rem;max-width:420px;margin:0 auto;line-height:1.7;">
Drop a YouTube link or a recording into the console on the left,
then hit <strong>Analyze</strong> — MinuteMind will find the signal in the noise.
</div>
<div style="margin-top:1.6rem;">
<span class="badge">Transcription</span>
<span class="badge">Digest</span>
<span class="badge">Live chat</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="channel-tag">Session</div>'
        f'<div class="channel-title" style="margin-bottom:1rem;">{esc(sanitize_content(st.session_state.title))}</div>',
        unsafe_allow_html=True,
    )

    channel_card("CH.01 — SUMMARY", "", st.session_state.summary)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        channel_card("CH.02 — ACTION ITEMS", "", st.session_state.action_items)
    with c2:
        channel_card("CH.03 — KEY DECISIONS", "", st.session_state.key_decisions)
    with c3:
        channel_card("CH.04 — OPEN QUESTIONS", "", st.session_state.open_questions)

    with st.expander("CH.05 — TRANSCRIPT"):
        st.markdown(
            f'<div class="channel-body mono" style="max-height:340px;overflow-y:auto;white-space:pre-wrap;">'
            f'{esc(sanitize_content(st.session_state.transcript))}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="channel-tag"><span class="led"></span>CH.06 — ASK MINUTEMIND</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history:
        rows = []
        for role, msg in st.session_state.chat_history:
            label = "You" if role == "user" else "MinuteMind"
            content = esc(msg) if role == "user" else render_body(msg)
            rows.append(
                f'<div class="msg {role}"><div class="msg-label">{label}</div>'
                f'<div class="bubble">{content}</div></div>'
            )
        st.markdown(f'<div class="chat-log">{"".join(rows)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:var(--text-muted);font-size:0.85rem;margin-bottom:0.8rem;">'
            "It already listened — ask it anything about this meeting.</div>",
            unsafe_allow_html=True,
        )

    q_col, btn_col = st.columns([5, 1], gap="small")
    with q_col:
        question = st.text_input(
            "Ask", placeholder="What did we decide about the launch date?", label_visibility="collapsed"
        )
    with btn_col:
        ask_clicked = st.button("Send", use_container_width=True)

    if ask_clicked and question.strip():
        with st.spinner("Listening back…"):
            try:
                answer = ask_questions(st.session_state.rag_chain, question.strip())
            except Exception as e:
                answer = f"Couldn't answer that — {e}"
        st.session_state.chat_history.append(("user", question.strip()))
        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()