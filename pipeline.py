"""Chinese video -> Khmer dub. Analyze (whisper + pitch) then Render (edge-tts + ffmpeg)."""
import asyncio, json, os, re, subprocess
from pathlib import Path

import numpy as np

DATA = Path(os.getenv("DATA_DIR", "/tmp/data"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Voice profiles: Edge TTS only has 2 Khmer voices, so age/gender is shaped with pitch + rate.
PROFILES = {
    "child":    dict(label="កូនក្មេង",        voice="km-KH-SreymomNeural", pitch="+45Hz", rate="+8%"),
    "teen_f":   dict(label="ជំទង់ស្រី",        voice="km-KH-SreymomNeural", pitch="+18Hz", rate="+4%"),
    "teen_m":   dict(label="ជំទង់ប្រុស",       voice="km-KH-PisethNeural",  pitch="+14Hz", rate="+4%"),
    "adult_f":  dict(label="ស្ត្រីវ័យកណ្តាល",   voice="km-KH-SreymomNeural", pitch="+0Hz",  rate="+0%"),
    "adult_m":  dict(label="បុរសវ័យកណ្តាល",    voice="km-KH-PisethNeural",  pitch="+0Hz",  rate="+0%"),
    "elder_f":  dict(label="ស្ត្រីចាស់",        voice="km-KH-SreymomNeural", pitch="-18Hz", rate="-14%"),
    "elder_m":  dict(label="បុរសចាស់",         voice="km-KH-PisethNeural",  pitch="-22Hz", rate="-14%"),
    "narrator": dict(label="អ្នកនិទាន",        voice="km-KH-PisethNeural",  pitch="-6Hz",  rate="-4%"),
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])
    return r.stdout


def probe_duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)]).strip())


# ---------- analyze ----------
_whisper = None

def transcribe(wav, language):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                                download_root=str(DATA / "whisper"))
    segs, _ = _whisper.transcribe(str(wav), language=None if language == "auto" else language,
                                  vad_filter=True, beam_size=5)
    out = []
    for s in segs:
        t = s.text.strip()
        if t:
            out.append(dict(start=round(s.start, 2), end=round(s.end, 2), zh=t))
    return out


def add_pitch(wav, segs):
    import librosa
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    for s in segs:
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        clip = y[a:b]
        s["f0"] = None
        if len(clip) < 2048:
            continue
        try:
            f0 = librosa.yin(clip, fmin=70, fmax=450, sr=sr, frame_length=2048)
            rms = librosa.feature.rms(y=clip, frame_length=2048, hop_length=512)[0]
            n = min(len(f0), len(rms))
            f0, rms = f0[:n], rms[:n]
            mask = rms > 0.6 * np.median(rms)
            if mask.sum() > 3:
                s["f0"] = int(np.median(f0[mask]))
        except Exception:
            pass


# ---- translate: local NLLB-200 model (no AI API, no internet needed after first download) ----
_nllb = None

def load_nllb():
    global _nllb
    if _nllb is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
        tok = AutoTokenizer.from_pretrained(name, src_lang="zho_Hans")
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
        torch.set_num_threads(os.cpu_count() or 2)
        _nllb = (tok, mdl)
    return _nllb


def translate(segs, progress):
    import torch
    tok, mdl = load_nllb()
    bos = tok.convert_tokens_to_ids("khm_Khmr")
    B = 8
    for i in range(0, len(segs), B):
        batch = segs[i:i + B]
        enc = tok([s["zh"] for s in batch], return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            out = mdl.generate(**enc, forced_bos_token_id=bos, num_beams=int(os.getenv("NLLB_BEAMS", "2")), max_new_tokens=256)
        for s, t in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            s["km"] = t.strip()
        progress(min(1.0, (i + B) / len(segs)))


# ---- speakers: cluster by pitch, then map pitch -> age/gender profile ----
def kmeans1d(x, k, iters=30):
    c = np.quantile(x, np.linspace(0.1, 0.9, k)) if k > 1 else np.array([np.median(x)])
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None]), axis=1)
        c = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
    return c, np.argmin(np.abs(x[:, None] - c[None]), axis=1)


def profile_for(f0, text):
    if re.search("老身|老夫|老朽|我老了|老婆子", text):
        return "elder_f" if re.search("老身|老婆子", text) else "elder_m"
    if f0 >= 290:
        return "child"
    return "adult_f" if f0 >= 185 else "adult_m"


def assign_speakers(segs):
    idx = [i for i, s in enumerate(segs) if s.get("f0")]
    lab_of = {}
    if len(idx) >= 4:
        x = np.array([segs[i]["f0"] for i in idx], dtype=float)
        for k in (4, 3, 2, 1):
            c, lab = kmeans1d(x, k)
            order = np.sort(c)
            sizes = np.bincount(lab, minlength=k)
            if (k == 1 or (np.diff(order).min() >= 30 and sizes.min() >= max(2, 0.08 * len(x)))):
                break
        rank = {int(np.argsort(c)[j]): j for j in range(k)}  # low pitch -> S1
        lab_of = {i: rank[int(l)] for i, l in zip(idx, lab)}
    last = 0
    for i, s in enumerate(segs):
        last = lab_of.get(i, last)
        s["speaker"] = f"S{last + 1}"
    speakers = {}
    for sid in sorted({s["speaker"] for s in segs}):
        mine = [s for s in segs if s["speaker"] == sid]
        f = [s["f0"] for s in mine if s.get("f0")]
        speakers[sid] = profile_for(float(np.median(f)) if f else 140, " ".join(s["zh"] for s in mine))
    return speakers


def analyze(job, save):
    d = DATA / "jobs" / job["id"]
    video, wav = d / "input.mp4", d / "speech16k.wav"

    def step(stage, p, msg):
        print(f"[{job['id']}] {stage} {int(p*100)}% {msg}", flush=True)
        job.update(stage=stage, progress=p, message=msg)
        save(job)

    step("audio", 0.05, "កំពុងទាញសំឡេងចេញពីវីដេអូ")
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    step("whisper", 0.15, "កំពុងស្តាប់ និងចម្លងសំឡេងចិន (Whisper)")
    segs = transcribe(wav, job["options"].get("language", "zh"))
    if not segs:
        raise RuntimeError("រកមិនឃើញសំឡេងនិយាយក្នុងវីដេអូនេះទេ")
    step("pitch", 0.45, "កំពុងវាស់កម្ពស់សំឡេងអ្នកនិយាយ")
    add_pitch(wav, segs)
    step("translate", 0.55, "កំពុងប្រែជាខ្មែរ (លើកដំបូងត្រូវទាញយក model ចាំបន្តិច)")
    translate(segs, lambda p: step("translate", 0.55 + 0.35 * p, "កំពុងប្រែជាខ្មែរ"))
    step("speakers", 0.92, "កំពុងបែងចែកអ្នកនិយាយតាមសំឡេង")
    speakers = assign_speakers(segs)
    job["segments"] = [dict(id=i, **s) for i, s in enumerate(segs)]
    job["speakers"] = {k: dict(profile=v) for k, v in speakers.items()}
    job.update(status="review", stage="review", progress=1.0, message="រួចរាល់ សូមពិនិត្យ និងកែ")
    save(job)


# ---------- render ----------
async def tts(text, profile, out, retries=4):
    import edge_tts
    p = PROFILES[profile]
    for a in range(retries):
        try:
            await edge_tts.Communicate(text, p["voice"], rate=p["rate"], pitch=p["pitch"]).save(str(out))
            return
        except Exception:
            if a == retries - 1:
                raise
            await asyncio.sleep(1.5 * (a + 1))


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def write_srt(segs, path):
    with open(path, "w", encoding="utf-8") as f:
        for n, s in enumerate(segs, 1):
            f.write(f"{n}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['km']}\n\n")


def fit_speed(src, dst, max_sec):
    from pydub import AudioSegment
    dur = len(AudioSegment.from_file(src)) / 1000
    if dur <= max_sec or max_sec <= 0.3:
        return src
    speed = min(dur / max_sec, 1.5)
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={speed:.3f}", str(dst)])
    return dst


async def synth_all(job, d):
    segs, sp = job["segments"], job["speakers"]
    sem = asyncio.Semaphore(4)
    done = 0

    async def one(i, s):
        nonlocal done
        out = d / "tts" / f"{i}.mp3"
        async with sem:
            await tts(s["km"], sp[s["speaker"]]["profile"], out)
        done += 1
        job.update(progress=0.1 + 0.6 * done / len(segs), message=f"កំពុងបង្កើតសំឡេង {done}/{len(segs)}")
        return out

    (d / "tts").mkdir(exist_ok=True)
    return await asyncio.gather(*[one(i, s) for i, s in enumerate(segs) if s["km"].strip()])


def render(job, save):
    from pydub import AudioSegment
    d = DATA / "jobs" / job["id"]
    opts = job["render"]
    segs = [s for s in job["segments"] if s["km"].strip()]
    job.update(status="rendering", stage="tts", progress=0.1, message="កំពុងបង្កើតសំឡេងខ្មែរ")
    save(job)
    asyncio.run(synth_all(job, d))

    job.update(stage="mix", progress=0.75, message="កំពុងលាយសំឡេង")
    run(["ffmpeg", "-y", "-i", str(d / "input.mp4"), "-vn", "-ac", "2", "-ar", "44100", str(d / "orig.wav")])
    base = AudioSegment.from_file(d / "orig.wav")
    bg = float(opts.get("bg", 0.12))
    base = base - 120 if bg <= 0 else base + 20 * np.log10(bg)
    track = AudioSegment.silent(duration=len(base), frame_rate=44100).set_channels(2)
    for k, s in enumerate(segs):
        nxt = segs[k + 1]["start"] if k + 1 < len(segs) else s["end"] + 3
        src = d / "tts" / f"{s['id']}.mp3"
        src = fit_speed(src, d / "tts" / f"{s['id']}_f.mp3", nxt - s["start"] - 0.05)
        clip = AudioSegment.from_file(src).set_frame_rate(44100).set_channels(2)
        track = track.overlay(clip, position=int(s["start"] * 1000))
    mix = base.overlay(track + 3)
    mix.export(d / "dub.mp3", format="mp3", bitrate="192k")
    mix.export(d / "mix.wav", format="wav")

    write_srt(segs, d / "subs.srt")
    job.update(stage="mux", progress=0.88, message="កំពុងភ្ជាប់ចូលវីដេអូ")
    save(job)
    out = d / "dubbed.mp4"
    if opts.get("burn"):
        style = "FontName=Noto Sans Khmer,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,MarginV=28"
        vf = f"subtitles={d / 'subs.srt'}:force_style='{style}'"
        run(["ffmpeg", "-y", "-i", str(d / "input.mp4"), "-i", str(d / "mix.wav"), "-map", "0:v:0", "-map", "1:a:0",
             "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "192k",
             "-shortest", str(out)])
    else:
        run(["ffmpeg", "-y", "-i", str(d / "input.mp4"), "-i", str(d / "mix.wav"), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)])
    job.update(status="done", stage="done", progress=1.0, message="រួចរាល់!")
    save(job)
