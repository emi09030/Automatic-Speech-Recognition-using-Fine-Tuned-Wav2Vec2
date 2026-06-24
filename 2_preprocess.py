import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

def speech_preprocessing_pipeline(audio_array, sr):
    y = np.array(audio_array, dtype=np.float32)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    y_segmented, _ = librosa.effects.trim(y, top_db=20)
    y_normalized = librosa.util.normalize(y_segmented)
    y_preemphasized = librosa.effects.preemphasis(y_normalized)
    
    return y_preemphasized, sr

def extract_and_save_all_features(y, sr, output_filename):
    os.makedirs("features_output", exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    
    axes[0, 0].plot(np.linspace(0, len(y)/sr, len(y)), y, color='blue')
    axes[0, 0].set_title("1. Waveform", fontsize=12, fontweight='bold')
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1], cmap='magma')
    axes[0, 1].set_title("2. Spectrogram", fontsize=12, fontweight='bold')
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title("3. MFCC", fontsize=12, fontweight='bold')
    
    rms = librosa.feature.rms(y=y)[0]
    t = librosa.frames_to_time(range(len(rms)), sr=sr)
    axes[1, 1].plot(t, rms, color='red')
    axes[1, 1].set_title("4. RMS Energy", fontsize=12, fontweight='bold')
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    axes[2, 0].plot(t, zcr, color='green')
    axes[2, 0].set_title("5. Zero Crossing Rate (ZCR)", fontsize=12, fontweight='bold')
    
    pitch_yin, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    axes[2, 1].plot(librosa.times_like(pitch_yin, sr=sr), pitch_yin, marker='o', linestyle='None', color='purple', markersize=2)
    axes[2, 1].set_title("6. Pitch Tracking (F0)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = f"features_output/{output_filename}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Thành công] Đã lưu đồ thị tại: {save_path}")

if __name__ == "__main__":
    # Lấy 1 mẫu ra để vẽ báo cáo nên chỉ cần load validation cho nhanh
    dataset = load_dataset("librispeech_asr", "clean", split="validation", trust_remote_code=True)
    raw_audio, sr = dataset[0]['audio']['array'], dataset[0]['audio']['sampling_rate']
    y_processed, sample_rate = speech_preprocessing_pipeline(raw_audio, sr)
    extract_and_save_all_features(y_processed, sample_rate, "LibriSpeech_Features")