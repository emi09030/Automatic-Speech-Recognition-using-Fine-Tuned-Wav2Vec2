import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

# Đăng nhập Hugging Face
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def explore_dataset():
    print("--- KHẢO SÁT VÀ TẢI DỮ LIỆU LIBRISPEECH ---")
    print("Đang kết nối Hugging Face tải dataset (Giới hạn 30.000 mẫu để chống tràn RAM)...")
    
    # CÚ PHÁP QUAN TRỌNG: Chỉ nạp 30.000 dòng từ tập train.360
    dataset = load_dataset("librispeech_asr", "clean", split="train.360[:30000]", trust_remote_code=True)
    
    print(f"\n[+] Tổng số mẫu đã tải thành công vào RAM: {len(dataset)}")
    
    sample = dataset[0]
    print("\n[+] Cấu trúc của một mẫu dữ liệu (Data Sample):")
    print(f"- ID: {sample['id']}")
    print(f"- Audio Array Shape: {sample['audio']['array'].shape}")
    print(f"- Sampling Rate: {sample['audio']['sampling_rate']} Hz")
    print(f"- Transcript: {sample['text']}")
    print(f"- Thời lượng: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f} giây")

if __name__ == "__main__":
    explore_dataset()