"""
Voice transcription using faster-whisper.
- Uses 'tiny' or 'base' model to minimize RAM/disk usage
- Runs on CPU (no GPU required)
- faster-whisper is significantly faster than openai-whisper on CPU
"""
import logging

logger = logging.getLogger(__name__)

# Lazy-load the model so it doesn't block startup
_model = None
MODEL_SIZE = "tiny"   # tiny ~75MB, base ~145MB, small ~465MB


def _get_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model: {MODEL_SIZE}")
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            logger.info("Whisper model loaded.")
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text.
    Returns the transcribed string.
    """
    model = _get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    logger.info(f"Detected language: {info.language} (prob={info.language_probability:.2f})")
    text = " ".join(segment.text for segment in segments).strip()
    if not text:
        raise ValueError("No speech detected in audio file.")
    return text