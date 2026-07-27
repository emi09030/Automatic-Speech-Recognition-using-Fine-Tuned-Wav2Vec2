import gradio as gr
import torch
import librosa
import numpy as np
import time
import jiwer
import os

# Xử lý import moviepy an toàn cho mọi phiên bản
try:
    from moviepy.editor import AudioFileClip
except ImportError:
    from moviepy import AudioFileClip

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 1. Load Model
model_path = "./Model_FineTuned"
print("Đang nạp não bộ AI...")
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

def custom_normalization(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val > 0 else signal

def process_file(file_path):
    """Hàm xử lý tách âm thanh từ audio hoặc video"""
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Tách âm thanh từ video
        audio_path = "temp_extracted.wav"
        video = AudioFileClip(file_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        return audio_path
    return file_path

def analyze(file_obj, reference_text):
    if file_obj is None:
        return "Vui lòng tải file!", 0, 0, 0
    
    start_time = time.time()
    
    # 1. Xử lý file đầu vào
    audio_path = process_file(file_obj)
    
    # 2. Load và Normalize
    speech, _ = librosa.load(audio_path, sr=16000)
    speech = custom_normalization(speech)
    
    # 3. Inference
    input_values = processor(speech, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()
    
    end_time = time.time()
    duration = round(end_time - start_time, 4)
    
    # 4. Tính toán Metrics (WER & CER)
    wer = 0
    cer = 0
    if reference_text:
        wer = round(jiwer.wer(reference_text, transcription), 4)
        cer = round(jiwer.cer(reference_text, transcription), 4)
    
    return transcription, wer, cer, duration

# 2. Giao diện Gradio Blocks (Chuyên nghiệp hơn)
with gr.Blocks(title="Nhóm 10 - ASR System") as demo:
    gr.Markdown("# Hệ thống nhận dạng tiếng nói ASR - Wav2Vec2")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Tải file âm thanh (wav)", type="filepath")
            ref_text = gr.Textbox(label="Văn bản gốc (Ground Truth) để so sánh", placeholder="Nhập văn bản gốc tại đây...")
            submit_btn = gr.Button("Bắt đầu phân tích")
        
        with gr.Column():
            transcript_out = gr.Textbox(label="Kết quả nhận dạng (Transcript)")
            wer_out = gr.Number(label="WER (Word Error Rate)")
            cer_out = gr.Number(label="CER (Character Error Rate)")
            time_out = gr.Number(label="Thời gian xử lý (giây)")

    submit_btn.click(
        fn=analyze, 
        inputs=[audio_input, ref_text], 
        outputs=[transcript_out, wer_out, cer_out, time_out]
    )

if __name__ == "__main__":
    demo.launch()