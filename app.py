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
MAX_MB = int(os.getenv("MAX_MB", "1000"))
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


TTL = float(os.getenv("JOB_TTL_HOURS", "12")) * 3600
ACTIVE = ("analyzing", "rendering", "queued")


def purge(force=False):
    """Delete old jobs (or all idle jobs when disk is low) so the volume never fills up."""
    import time
    now = time.time()
    for d in JOBS_DIR.iterdir():
        j = JOBS.get(d.name)
        if j and j.get("status") in ACTIVE:
            continue
        if not (d / "job.json").exists() and now - d.stat().st_mtime > 600:
            shutil.rmtree(d, ignore_errors=True)  # orphan from a failed upload
            continue
        if force or now - d.stat().st_mtime > TTL:
            shutil.rmtree(d, ignore_errors=True)
            JOBS.pop(d.name, None)
    for f in DATA.glob("prev_*.mp3"):
        f.unlink(missing_ok=True)


def low_disk():
    return shutil.disk_usage(DATA).free < 1.5 * 1024**3


def clean_models():
    """Remove cached models we no longer use (e.g. a big NLLB variant that was tried before)."""
    keep = "models--" + os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M").replace("/", "--")
    hub = DATA / "hf" / "hub"
    if hub.exists():
        for d in hub.glob("models--*"):
            if d.name != keep:
                shutil.rmtree(d, ignore_errors=True)
                print("removed unused model cache", d.name, flush=True)


clean_models()
purge()


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
    purge(force=low_disk())
    jid = uuid.uuid4().hex[:10]
    d = JOBS_DIR / jid
    try:
        d.mkdir(parents=True)
        with open(d / "input.mp4", "wb") as f:
            size = 0
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024:
                    raise HTTPException(413, f"ឯកសារធំពេក (លើស {MAX_MB} MB) សូមបង្រួមវីដេអូមុន")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    except Exception as e:
        traceback.print_exc()
        shutil.rmtree(d, ignore_errors=True)
        msg = "ទំហំ Volume ពេញ សូមបង្កើនទំហំ Volume ក្នុង Railway" if "No space" in str(e) else str(e)
        raise HTTPException(500, f"រក្សាទុកឯកសារមិនបាន: {msg}")
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


@app.get("/api/disk")
def disk():
    def size(p):
        return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    u = shutil.disk_usage(DATA)
    out = {"total_gb": round(u.total / 1e9, 2), "free_gb": round(u.free / 1e9, 2),
           "NLLB_MODEL": os.getenv("NLLB_MODEL", "(default 600M)"), "WHISPER_MODEL": pipeline.WHISPER_MODEL}
    for d in sorted(DATA.iterdir()):
        out[d.name] = round((size(d) if d.is_dir() else d.stat().st_size) / 1e9, 3)
    return out


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
