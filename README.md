# ClipMaker AI — Free starter

A starter MVP for a Vizard-style clipping website. It accepts a video URL, downloads it with yt-dlp, transcribes speech with the open-source faster-whisper `tiny` model, heuristically selects speech-dense moments, and exports vertical 9:16 MP4 clips with burned-in subtitles.

## Run locally

Requirements: Python 3.10+, FFmpeg installed and available on PATH.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

The first request downloads the Whisper tiny model. CPU processing can be slow.

## Free hosting

This project can be tested locally for free. Public hosting of video processing is harder: free hosts often sleep, have limited RAM/CPU, temporary storage, and request timeouts. The Whisper model and FFmpeg also need server resources. Do not promise unlimited free processing.

## Important

- Use only videos you own or have permission to process. Respect YouTube's terms, copyright, and privacy rights.
- The current AI clip selection is a lightweight prototype based on transcript density and cue words, not the same quality as commercial highlight-detection systems.
- For production, add job queues, progress updates, storage limits, user authentication, abuse protection, cleanup, and a proper rights/consent flow.
- Never expose private API keys in browser JavaScript.
