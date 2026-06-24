import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer, cer

MODEL_VERSION = "v0_baseline"
LOCAL_SAVE_PATH = f"models_saved/{MODEL_VERSION}"
HF_MODEL_ID = "facebook/wav2vec2-base-960h"

def evaluate_baseline_system():
    print(f"--- TẢI VÀ LƯU MÔ HÌNH BASELINE ({MODEL_VERSION}) ---")
    processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)
    
    os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)
    processor.save_pretrained(LOCAL_SAVE_PATH)
    model.save_pretrained(LOCAL_SAVE_PATH)
    print(f"-> Mô hình đã lưu Offline tại: {LOCAL_SAVE_PATH}")
    
    print("\n--- INFERENCE VÀ ĐÁNH GIÁ WER/CER ---")
    # Đánh giá luôn chạy trên tập validation hoặc test (chỉ khoảng 2700 mẫu, rất nhẹ)
    dataset = load_dataset("librispeech_asr", "clean", split="validation", trust_remote_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    for i in range(10): # Đánh giá thử 10 câu cho báo cáo giữa kỳ
        sample = dataset[i]
        audio_array = sample['audio']['array']
        ground_truth = sample['text'].upper()
        
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0].upper()
            
        results.append({
            "Mẫu": i+1,
            "Thực tế": ground_truth,
            "Dự đoán": transcription,
            "WER (%)": round(wer(ground_truth, transcription) * 100, 2),
            "CER (%)": round(cer(ground_truth, transcription) * 100, 2)
        })
        
    df = pd.DataFrame(results)
    print("\nBẢNG BÁO CÁO SAI SỐ:")
    print(df.to_string(index=False))
    df.to_csv(f"models_saved/{MODEL_VERSION}_Evaluation.csv", index=False)

if __name__ == "__main__":
    evaluate_baseline_system()