import os, re, uuid, shutil, subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
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

class ClipRequest(BaseModel):
    url: HttpUrl
    count: int = 3
    duration: int = 30

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2500:])
    return p.stdout

def get_model():
    # First run downloads the small open-source model; CPU mode works without an API key.
    return WhisperModel("tiny", device="cpu", compute_type="int8")

@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")

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
            info = ydl.extract_info(str(req.url), download=True)
            source = Path(ydl.prepare_filename(info))
        if not source.exists():
            candidates = list(folder.glob("source.*"))
            if not candidates:
                raise RuntimeError("ভিডিও ডাউনলোড করা যায়নি।")
            source = candidates[0]

        model = get_model()
        segments, _ = model.transcribe(str(source), vad_filter=True, word_timestamps=False)
        transcript = []
        for s in segments:
            txt = re.sub(r"\s+", " ", s.text).strip()
            if txt:
                transcript.append({"start": float(s.start), "end": float(s.end), "text": txt})

        if not transcript:
            raise RuntimeError("ভিডিওতে শনাক্ত করার মতো বক্তব্য পাওয়া যায়নি।")

        # Score overlapping time windows using speech density and useful sentence cues.
        total = max(x["end"] for x in transcript)
        duration = max(15, min(int(req.duration), 60))
        candidates = []
        step = max(5, duration // 3)
        for start in range(0, max(1, int(total - duration) + 1), step):
            end = min(total, start + duration)
            items = [x for x in transcript if x["end"] > start and x["start"] < end]
            text = " ".join(x["text"] for x in items)
            words = len(text.split())
            if words < 12:
                continue
            cue = sum(text.lower().count(k) for k in
                      ["because", "but", "so", "why", "how", "important", "সত্যি", "কারণ", "কিন্তু", "কেন", "কিভাবে"])
            score = words + cue * 4
            candidates.append((score, start, end, text))
        candidates.sort(reverse=True)
        chosen = []
        for item in candidates:
            _, start, end, _ = item
            if all(abs(start - old[1]) >= duration * 0.65 for old in chosen):
                chosen.append(item)
            if len(chosen) >= max(1, min(req.count, 5)):
                break
        if not chosen:
            chosen = [(0, 0, min(total, duration), "")]
        results = []
        for i, (_, start, end, caption) in enumerate(chosen, 1):
            out = folder / f"clip_{i}.mp4"
            # Center-crop to vertical 9:16, add readable burned-in subtitles.
            subtitle = folder / f"clip_{i}.srt"
            def stamp(t):
                ms = int(max(0, t) * 1000)
                h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"
            lines = []
            for seg in transcript:
                a, b = max(start, seg["start"]), min(end, seg["end"])
                if b > a:
                    lines.append(f"{len(lines)+1}\n{stamp(a-start)} --> {stamp(b-start)}\n{seg['text']}\n")
            subtitle.write_text("\n".join(lines), encoding="utf-8")
            vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
            # Subtitle path is escaped for ffmpeg filter syntax.
            subpath = str(subtitle).replace("\\", "/").replace(":", r"\:")
            vf += f",subtitles='{subpath}':force_style='FontSize=18,Outline=2,Alignment=2,MarginV=140'"
            run(["ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(end-start),
                 "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
            results.append({"name": f"Clip {i}", "url": f"/files/{job}/{out.name}",
                            "start": round(start), "duration": round(end-start),
                            "transcript": caption[:220]})
        return {"job": job, "clips": results,
                "note": "AI selection is an initial prototype; review clips before publishing."}
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))

