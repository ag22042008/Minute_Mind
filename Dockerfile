# ─────────────────────────────────────────────────────────────────────────────
# MinuteMind — Docker Image
#
# Key differences from Streamlit Cloud that make YouTube audio download work:
#   1. Node.js 22 LTS  — modern enough to solve yt-dlp's n-challenge
#   2. Deno            — alternative JS runtime, better n-challenge support
#   3. VPS IP address  — YouTube doesn't block residential/VPS IPs with PO tokens
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 22 (LTS) via NodeSource ───────────────────────────────────────────
# Ubuntu/Debian default nodejs is 12.x — too old to solve YouTube's n-challenge.
# Node.js 22 is current LTS and handles YouTube's obfuscated JS correctly.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version

# ── Deno (alternative JS runtime) ─────────────────────────────────────────────
# yt-dlp prefers Deno over Node.js for n-challenge if available.
# Deno is a more modern runtime with better ES module support.
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh
ENV PATH="/usr/local/bin:$PATH"

# ── Python app setup ───────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create downloads directory
RUN mkdir -p downloades

# ── Streamlit configuration ───────────────────────────────────────────────────
# Prevent Streamlit from opening a browser and showing usage stats
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
