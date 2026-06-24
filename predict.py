import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

audio_path = sys.argv[1]
MODEL_PATH = "models_saved/v0_baseline"

try:
    device = torch.device("cpu")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(device)

    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # In ra để Node.js bắt kết quả
    print(transcription)
except Exception as e:
    print(f"Error AI: {str(e)}")