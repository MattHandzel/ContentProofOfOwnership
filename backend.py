import os
import io
import shutil
import tempfile
import logging
import zipfile
from typing import List

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from audioseal import AudioSeal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_watermark_server")

app = FastAPI()

# Allow all CORS origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model Loading (at startup)
# =============================================================================
logger.info("Loading watermarking and detection models...")
# Load the watermarking generator model
watermark_model = AudioSeal.load_generator("audioseal_wm_16bits")
# Load the detector model to check if audio is watermarked
detector_model = AudioSeal.load_detector("audioseal_detector_16bits")
logger.info("Models loaded successfully.")


# =============================================================================
# Utility Functions
# =============================================================================
def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """
    Save an UploadFile to a temporary file and return its path.
    """
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = upload_file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(
            f"Saved upload file '{upload_file.filename}' to temporary path: {tmp_path}"
        )
        return tmp_path
    except Exception as e:
        logger.error(f"Error saving uploaded file '{upload_file.filename}': {e}")
        raise


def remove_temp_file(file_path: str):
    """
    Remove the temporary file.
    """
    try:
        os.remove(file_path)
        logger.info(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error removing temporary file '{file_path}': {e}")


# =============================================================================
# Core Functions
# =============================================================================
def watermark_audios(paths: List[str], model) -> List[str]:
    """
    For each file path in 'paths', load the audio, resample if needed,
    compute the watermark, add it to the audio, and save the watermarked audio
    as a new temporary file. Returns a list of paths to the watermarked files.
    """
    watermarked_paths = []
    for audio_path in paths:
        try:
            logger.info(f"Watermarking file: {audio_path}")
            # Load audio file
            wav, sr = torchaudio.load(audio_path)
            # Resample if necessary
            if sr != 16000:
                logger.info(f"Resampling audio from {sr} Hz to 16000 Hz: {audio_path}")
                transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = transform(wav)
                sr = 16000

            # Add batch dimension (expected shape: batch, channels, samples)
            wav = wav.unsqueeze(0)
            # Get watermark and add to the original audio
            watermark = model.get_watermark(wav, sr)
            watermarked_audio = wav + watermark

            # Save watermarked audio to a temporary file
            watermarked_tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix="_watermarked.wav"
            )
            torchaudio.save(watermarked_tmp.name, watermarked_audio.squeeze(0), sr)
            watermarked_paths.append(watermarked_tmp.name)
            logger.info(f"Saved watermarked file: {watermarked_tmp.name}")
        except Exception as e:
            logger.error(f"Error processing file '{audio_path}': {e}")
    return watermarked_paths


def detect_watermarked_audios(paths: List[str], detector) -> List[dict]:
    """
    For each audio file in paths, load the file, resample if necessary,
    and use the detector to determine if the file is watermarked. Returns a
    list of dictionaries with the detection results.
    """
    results = []
    for audio_path in paths:
        try:
            logger.info(f"Detecting watermark in file: {audio_path}")
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                logger.info(f"Resampling audio from {sr} Hz to 16000 Hz: {audio_path}")
                transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                # If stereo, take the mean to get a single channel
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                wav = transform(wav)
                sr = 16000

            wav = wav.unsqueeze(0)  # Add batch dimension
            result, message = detector.detect_watermark(wav, sr)
            results.append(
                {
                    "file": os.path.basename(audio_path),
                    "watermark_probability": result.tolist(),
                    "watermark_message": message.tolist(),
                }
            )
            logger.info(f"Detection completed for file: {audio_path}")
        except Exception as e:
            logger.error(f"Error detecting watermark in file '{audio_path}': {e}")
            results.append({"file": os.path.basename(audio_path), "error": str(e)})
    return results


# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/watermark")
async def watermark_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more audio files to be watermarked.
    Returns a single watermarked file if one file is uploaded,
    or a zipped archive of watermarked files if multiple files are uploaded.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    temp_paths = []
    watermarked_paths = []
    try:
        # Save each uploaded file to a temporary location
        for file in files:
            tmp_path = save_upload_file_tmp(file)
            temp_paths.append(tmp_path)

        # Process watermarking
        watermarked_paths = watermark_audios(temp_paths, watermark_model)

        # If only one file was uploaded, return the watermarked file directly
        if len(watermarked_paths) == 1:
            file_path = watermarked_paths[0]
            logger.info(f"Returning single watermarked file: {file_path}")
            return StreamingResponse(
                open(file_path, "rb"),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
                },
            )
        else:
            # Zip multiple watermarked files for download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file_path in watermarked_paths:
                    zip_file.write(file_path, arcname=os.path.basename(file_path))
            zip_buffer.seek(0)
            logger.info("Returning zipped archive of watermarked files")
            return StreamingResponse(
                zip_buffer,
                media_type="application/x-zip-compressed",
                headers={
                    "Content-Disposition": "attachment; filename=watermarked_audios.zip"
                },
            )
    except Exception as e:
        logger.error(f"Error in /watermark endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up all temporary files created during processing
        for path in temp_paths:
            try:
                remove_temp_file(path)
            except Exception:
                pass
        for path in watermarked_paths:
            try:
                remove_temp_file(path)
            except Exception:
                pass


@app.post("/detect")
async def detect_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more audio files to detect if they have been watermarked.
    Returns detection results including watermark probability and message.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    temp_paths = []
    try:
        for file in files:
            # TODO: Downlaod audios if they don't already exist
            tmp_path = save_upload_file_tmp(file)
            temp_paths.append(tmp_path)

        detection_results = detect_watermarked_audios(temp_paths, detector_model)
        logger.info("Watermark detection completed for all files")
        return JSONResponse(content={"results": detection_results})
    except Exception as e:
        logger.error(f"Error in /detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in temp_paths:
            try:
                remove_temp_file(path)
            except Exception:
                pass


# =============================================================================
# Run the Server
# =============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
