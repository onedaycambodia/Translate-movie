import json, os, shutil, threading, traceback, uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

import pipeline
from pipeline import DATA, PROFILES

MAX_SECONDS = int(os.getenv("MAX_SECONDS", "600"))
JOBS_DIR = DATA / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Khmer Dubber")
LOCK = threading.Lock()  # one heavy job at a time (protects RAM)
JOBS = {}


def save(job):
    (JOBS_DIR / job["id"] / "job.json").write_text(json.dumps(job, ensure_ascii=False))


for p in JOBS_DIR.glob("*/job.json"):
    try:
        j = json.loads(p.read_text())
        if j.get("status") in ("queued", "analyzing", "rendering"):
            j.update(status="error", error="Server restarted. សូម upload ម្តងទៀត")
        JOBS[j["id"]] = j
    except Exception:
        pass


def get(jid):
    if jid not in JOBS:
        raise HTTPException(404, "រកមិនឃើញការងារ")
    return JOBS[jid]


def worker(job, fn):
    with LOCK:
        try:
            fn(job, save)
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", error=str(e)[:500])
            save(job)


@app.get("/api/profiles")
def profiles():
    return {k: v["label"] for k, v in PROFILES.items()}


@app.post("/api/jobs")
async def create(file: UploadFile = File(...), language: str = Form("zh")):
    jid = uuid.uuid4().hex[:10]
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    with open(d / "input.mp4", "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    try:
        dur = pipeline.probe_duration(d / "input.mp4")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "ឯកសារនេះមិនមែនជាវីដេអូទេ")
    if dur > MAX_SECONDS:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"វីដេអូវែងពេក (អតិបរមា {MAX_SECONDS // 60} នាទី)")
    job = dict(id=jid, filename=file.filename, duration=dur, status="analyzing", stage="audio", progress=0.02,
               message="កំពុងចាប់ផ្តើម", error=None, options=dict(language=language), segments=[], speakers={})
    JOBS[jid] = job
    save(job)
    threading.Thread(target=worker, args=(job, pipeline.analyze), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def status(jid: str):
    return get(jid)


class Edit(BaseModel):
    segments: list
    speakers: dict


@app.put("/api/jobs/{jid}")
def edit(jid: str, body: Edit):
    job = get(jid)
    job["segments"], job["speakers"] = body.segments, body.speakers
    save(job)
    return {"ok": True}


class Preview(BaseModel):
    text: str
    profile: str


@app.post("/api/preview")
async def preview(body: Preview):
    if body.profile not in PROFILES or not body.text.strip():
        raise HTTPException(400, "bad request")
    out = DATA / f"prev_{uuid.uuid4().hex[:8]}.mp3"
    await pipeline.tts(body.text, body.profile, out)
    return FileResponse(out, media_type="audio/mpeg", background=BackgroundTask(os.remove, out))


class RenderReq(BaseModel):
    bg: float = 0.12
    burn: bool = False


@app.post("/api/jobs/{jid}/render")
def render(jid: str, body: RenderReq):
    job = get(jid)
    if job["status"] not in ("review", "done", "error") or not job["segments"]:
        raise HTTPException(400, "មិនទាន់ត្រៀមរួច")
    job.update(render=dict(bg=body.bg, burn=body.burn), status="rendering", error=None)
    threading.Thread(target=worker, args=(job, pipeline.render), daemon=True).start()
    return {"ok": True}


@app.get("/api/jobs/{jid}/file/{kind}")
def file(jid: str, kind: str, dl: int = 0):
    get(jid)
    names = {"input": ("input.mp4", "video/mp4"), "video": ("dubbed.mp4", "video/mp4"),
             "srt": ("subs.srt", "text/plain"), "mp3": ("dub.mp3", "audio/mpeg")}
    if kind not in names or not (JOBS_DIR / jid / names[kind][0]).exists():
        raise HTTPException(404, "មិនមានឯកសារ")
    n, mt = names[kind]
    return FileResponse(JOBS_DIR / jid / n, media_type=mt, filename=f"khmer-dub-{jid}-{n}" if dl else None)


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
