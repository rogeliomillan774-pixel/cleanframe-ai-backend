import os, json, shutil, subprocess, tempfile, time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from rapidocr_onnxruntime import RapidOCR

APP_NAME = "CleanFrame AI Free Backend"
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "30"))
MAX_SIDE = int(os.getenv("MAX_FRAME_SIDE", "1280"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr

@app.get("/")
def root():
    return {"name": APP_NAME, "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "engine": "RapidOCR + OpenCV inpainting", "mode": ["auto", "manual"]}

def detect_text_mask(frame, ocr):
    result, _ = ocr(frame)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if result:
        for item in result:
            # RapidOCR returns [polygon, text, confidence]
            polygon = np.asarray(item[0], dtype=np.float32).astype(np.int32)
            if polygon.ndim == 2 and polygon.shape[0] >= 4:
                cv2.fillPoly(mask, [polygon], 255)
    # Expand slightly so text edges and anti-aliasing are included.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    return cv2.dilate(mask, kernel, iterations=1)

def manual_mask(shape, regions):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        try:
            x = max(0, min(w - 1, int(float(r["x"]) * w)))
            y = max(0, min(h - 1, int(float(r["y"]) * h)))
            rw = max(1, min(w - x, int(float(r["width"]) * w)))
            rh = max(1, min(h - y, int(float(r["height"]) * h)))
            cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, -1)
        except Exception:
            continue
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    return cv2.dilate(mask, kernel, iterations=1)

def run_ffmpeg_mux(original, silent_video, output, preserve_audio):
    if preserve_audio:
        cmd = [
            "ffmpeg", "-y", "-i", silent_video, "-i", original,
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", output
        ]
    else:
        cmd = ["ffmpeg", "-y", "-i", silent_video, "-c:v", "copy", "-movflags", "+faststart", output]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("FFmpeg export failed: " + p.stderr[-1200:])

@app.post("/api/remove-text")
async def remove_text(
    video: UploadFile = File(...),
    mode: str = Form("auto"),
    remove_all_text: bool = Form(True),
    preserve_audio: bool = Form(True),
    keep_resolution: bool = Form(True),
    manual_regions: str = Form("[]"),
):
    if mode not in ("auto", "manual"):
        raise HTTPException(400, "mode must be auto or manual")
    if not remove_all_text and mode == "auto":
        raise HTTPException(400, "Enable remove_all_text for automatic mode")
    try:
        regions = json.loads(manual_regions or "[]")
        if not isinstance(regions, list):
            raise ValueError()
    except Exception:
        raise HTTPException(400, "manual_regions must be a JSON array")

    work = tempfile.mkdtemp(prefix="cleanframe_")
    source = os.path.join(work, "source")
    silent = os.path.join(work, "silent.mp4")
    output = os.path.join(work, "cleanframe_cleaned.mp4")
    try:
        size = 0
        with open(source, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"Video exceeds {MAX_BYTES // (1024*1024)} MB limit")
                f.write(chunk)
        if size == 0:
            raise HTTPException(400, "No video file received")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise HTTPException(400, "Could not read video. Try MP4, MOV, or WEBM.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total / fps if total and fps else 0
        if duration > MAX_SECONDS:
            cap.release()
            raise HTTPException(413, f"Free demo limit is {MAX_SECONDS} seconds per video")
        if width < 2 or height < 2:
            cap.release()
            raise HTTPException(400, "Invalid video dimensions")

        scale = min(1.0, MAX_SIDE / max(width, height)) if not keep_resolution else 1.0
        out_w, out_h = int(width * scale), int(height * scale)
        out_w -= out_w % 2
        out_h -= out_h % 2
        writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise HTTPException(500, "Could not start video encoder")

        ocr = get_ocr() if mode == "auto" else None
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if mode == "auto":
                mask = detect_text_mask(frame, ocr)
            else:
                mask = manual_mask(frame.shape, regions)
            if np.any(mask):
                # Fast classical inpainting. This is a free baseline, not temporal AI inpainting.
                frame = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
            writer.write(frame)
            processed += 1
        cap.release()
        writer.release()
        if processed == 0:
            raise HTTPException(400, "No readable video frames found")

        run_ffmpeg_mux(source, silent, output, preserve_audio)
        return FileResponse(output, media_type="video/mp4", filename="cleanframe-cleaned.mp4",
                            background=BackgroundTask(shutil.rmtree, work, ignore_errors=True))
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Processing failed: {str(e)[:900]}")
