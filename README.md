# Khmer Dubber (ស្ទូឌីយ៉ូដាក់សំឡេងខ្មែរ)

Upload a Chinese video -> Whisper transcribes -> local NLLB-200 model translates to Khmer (no AI API) and speakers are split/classified by measured pitch -> you review/edit -> Edge TTS voices (child / teen / adult / elder) -> ffmpeg mixes it back into the video.

## Deploy on Railway
1. Push this folder to GitHub, then Railway -> New Project -> Deploy from GitHub repo (Dockerfile is detected).
2. Variables: No API key needed. Optional: `WHISPER_MODEL` (small|base|medium), `NLLB_MODEL` (default facebook/nllb-200-distilled-600M), `MAX_SECONDS` (default 600).
3. Add a Volume mounted at `/data` (keeps whisper model + jobs). Recommend >= 4 GB RAM.
4. Settings -> Networking -> Generate Domain.
