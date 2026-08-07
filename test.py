from utils.audio_processor import process_input
from core.transcriber import translate_all_sarvam

source = "https://youtu.be/e-4PU6vizCk?si=PJ18iv5pjbszKdkg"

chunks = process_input(source)

print(translate_all_sarvam(chunks))