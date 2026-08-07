import whisper
import os
import requests
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():

    global _model

    if _model is None:
        print(f"loading model ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("whisper model loaded successfully")

    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(chunk_path, task=task)

    return result['text']


def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"transcribing chunk{i+1}")
        text = transcribe_chunk(chunk, translate=translate)

        full_transcript += text + " "
    print("Transcription completed")

    return full_transcript


def translate_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not set in environment variables")

    headers = {
        "API-Subscription-Key": SARVAM_API_KEY
    }

    data = {
        "model": SARVAM_MODEL
    }

    with open(chunk_path, "rb") as f:
        files = {
            "file": (os.path.basename(chunk_path), f, "audio/wav")
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            data=data,
            files=files
        )

    if response.status_code != 200:
        raise Exception(f"Sarvam API error {response.status_code}: {response.text}")

    result = response.json()

    return result.get("transcript", "")


def translate_all_sarvam(chunks: list) -> str:
    full_translation = ""

    for i, chunk in enumerate(chunks):
        print(f"translating chunk{i+1} using Sarvam")
        text = translate_chunk_sarvam(chunk)

        full_translation += text + " "
    print("Sarvam translation completed")

    return full_translation