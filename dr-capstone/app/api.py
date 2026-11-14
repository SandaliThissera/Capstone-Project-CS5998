"""FastAPI inference service.  Run from the repo root:  uvicorn app.api:app --port 8000"""
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predictor import Predictor

state = {}


@asynccontextmanager
async def lifespan(app):
    state["p"] = Predictor(os.environ.get("DR_WEIGHTS", "export/ensemble.pt"))
    yield


app = FastAPI(title="DR grading prototype (research use only)", lifespan=lifespan)


@app.get("/health")
def health():
    p = state["p"]
    return {"status": "ok", "n_models": len(p.models), "cutoff": p.cutoff, "runs": p.runs}


@app.post("/predict")
def predict(file: UploadFile = File(...), cam: bool = True):
    arr = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "Could not decode the image (use PNG or JPEG).")
    if min(arr.shape[:2]) < 200:
        raise HTTPException(400, "Image is too small to be a fundus photograph.")
    return state["p"].predict(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), with_cam=cam)
