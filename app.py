
import os
import re
import uuid
import shutil
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
import yt_dlp
from faster_whisper import WhisperModel

BASE = Path(__file__).parent
WORK = BASE / "work"
WORK.mkdir(exist_ok=True)

app = FastAPI(title="ClipMaker AI")

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/files", StaticFiles(directory=WORK), name="files")

MAX_UPLOAD = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v"
}

_model = None
_model_lock = threading.Lock()


class ClipRequest(BaseModel):
    url: HttpUrl
    count: int = 3
    duration: int = 30


def run(cmd):
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600
    )
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout


def get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(
                    "tiny",
                    device="cpu",
                    compute_type="int8"
                )

    return _model


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ClipMaker AI"}


def create_clips(source, folder, count=3, duration=30):
    source = Path(source)

    if not source.exists():
        raise RuntimeError("ভিডিও ফাইল পাওয়া যায়নি।")

    model = get_model()

    segments, _ = model.transcribe(
        str(source),
        vad_filter=True,
        word_timestamps=False
    )

    transcript = []

    for s in segments:
        txt = re.sub(r"\s+", " ", s.text).strip()
        if txt:
            transcript.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": txt
            })

    if not transcript:
        raise RuntimeError("ভিডিওতে শনাক্ত করার মতো বক্তব্য পাওয়া যায়নি।")

    total = max(x["end"] for x in transcript)
    duration = max(15, min(int(duration), 60))
    count = max(1, min(int(count), 5))

    candidates = []
    step = max(5, duration // 3)

    for start in range(
        0,
        max(1, int(total - duration) + 1),
        step
    ):
        end = min(total, start + duration)

        items = [
            x for x in transcript
            if x["end"] > start and x["start"] < end
        ]

        text = " ".join(x["text"] for x in items)
        words = len(text.split())

        if words < 12:
            continue

        cues = [
            "because", "but", "so", "why", "how",
            "important", "সত্যি", "কারণ", "কিন্তু",
            "কেন", "কিভাবে"
        ]

        cue_score = sum(
            text.lower().count(k) for k in cues
        )

        score = words + cue_score * 4

        candidates.append((score, start, end, text))

    candidates.sort(reverse=True)

    chosen = []

    for item in candidates:
        _, start, end, caption = item

        if all(
            abs(start - old[1]) >= duration * 0.65
            for old in chosen
        ):
            chosen.append(item)

        if len(chosen) >= count:
            break

    if not chosen:
        chosen = [(0, 0, min(total, duration), "")]

    results = []

    def stamp(t):
        ms = int(max(0, t) * 1000)
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    for i, (_, start, end, caption) in enumerate(chosen, 1):

        out = folder / f"clip_{i}.mp4"
        subtitle = folder / f"clip_{i}.srt"

        lines = []

        for seg in transcript:
            a = max(start, seg["start"])
            b = min(end, seg["end"])

            if b > a:
                lines.append(
                    f"{len(lines)+1}\n"
                    f"{stamp(a-start)} --> {stamp(b-start)}\n"
                    f"{seg['text']}\n"
                )

        subtitle.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        vf = (
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )

        subpath = (
            str(subtitle)
            .replace("\\", "/")
            .replace(":", r"\:")
            .replace("'", r"\'")
        )

        vf += (
            f",subtitles='{subpath}':"
            "force_style='FontSize=18,"
            "Outline=2,Alignment=2,MarginV=140'"
        )

        run([
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(source),
            "-t", str(end - start),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "25",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(out)
        ])

        results.append({
            "name": f"Clip {i}",
            "url": f"/files/{folder.name}/{out.name}",
            "start": round(start),
            "duration": round(end-start),
            "transcript": caption[:220]
        })

    return {
        "job": folder.name,
        "clips": results,
        "note": "Review clips before publishing."
    }


@app.post("/api/clips")
def make_clips(req: ClipRequest):
    job = uuid.uuid4().hex[:10]
    folder = WORK / job
    folder.mkdir(parents=True)

    try:
        opts = {
            "format": "best[height<=720]/best",
            "outtmpl": str(folder / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "restrictfilenames": True,
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                str(req.url),
                download=True
            )
            source = Path(ydl.prepare_filename(info))

        if not source.exists():
            candidates = list(folder.glob("source.*"))

            if not candidates:
                raise RuntimeError("ভিডিও ডাউনলোড করা যায়নি।")

            source = candidates[0]

        return create_clips(
            source,
            folder,
            req.count,
            req.duration
        )

    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/clips/upload")
async def upload_video(
    file: UploadFile = File(...),
    count: int = Form(3),
    duration: int = Form(30)
):
    job = uuid.uuid4().hex[:10]
    folder = WORK / job
    folder.mkdir(parents=True)

    try:
        extension = Path(file.filename or "").suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="MP4, MOV, MKV, WEBM অথবা M4V ভিডিও দিন।"
            )

        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail="একটি বৈধ ভিডিও ফাইল Upload করুন।"
            )

        if not 1 <= count <= 5:
            raise HTTPException(
                status_code=400,
                detail="Clip সংখ্যা 1 থেকে 5-এর মধ্যে দিন।"
            )

        if not 15 <= duration <= 60:
            raise HTTPException(
                status_code=400,
                detail="Clip Duration 15 থেকে 60 সেকেন্ডের মধ্যে দিন।"
            )

        source = folder / ("source" + extension)
        size = 0

        with source.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_UPLOAD:
                    raise HTTPException(
                        status_code=413,
                        detail="সর্বোচ্চ 500 MB ভিডিও Upload করা যাবে।"
                    )

                destination.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="খালি ভিডিও ফাইল Upload করা যাবে না।"
            )

        return create_clips(
            source,
            folder,
            count,
            duration
        )

    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:
        await file.close()
