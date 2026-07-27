# Automatic Speech Recognition using Fine-Tuned Wav2Vec2
**with Data Augmentation and Language Model Rescoring for English Audio**

## 1. Project Overview
The explosion of online learning platforms, video conferences, and podcasts has generated a massive amount of audio and video data. Extracting and converting speech from these data sources into text is a crucial task for storing, searching, and analyzing information. This project focuses on building an **Automatic Speech Recognition (ASR)** system for English using the **Wav2Vec2** model. The system not only utilizes a pretrained model but also performs **Fine-tuning** on a standard dataset (LibriSpeech), combined with advanced techniques such as **Noise Reduction, Data Augmentation**, and **Language Model Rescoring** to optimize accuracy.

### Objectives
- Build an ASR system to extract text from Audio/Video files (English).
- Apply signal processing techniques (Segmentation, Normalization, Noise Reduction).
- Fine-tune Wav2Vec2 and compare performance across enhancement techniques.
- Provide a demo user interface (UI) for users to upload files and receive text results (Transcript).

---

## 2. System Pipeline

The system's data processing flow is designed with the following steps:

```text
[Input: Audio / Video]
        │
        ▼
[Audio Extraction] (Extract audio from video if applicable)
        │
        ▼
[Preprocessing] (Segmentation, Normalization)
        │
        ▼
[Noise Reduction] (Denoise signal)
        │
        ▼
[Data Augmentation] (Applied during Training Phase only)
        │
        ▼
[Wav2Vec2 Fine-tuned Model] (Main recognition model)
        │
        ▼
[Language Model Rescoring] (Optimize output text grammar)
        │
        ▼
[Output: Transcript]
```

---

## 3. Data & Evaluation

### Input & Output
- **Input:**
  - Audio files: `.wav`, `.mp3`, `.m4a`
  - Language: English
- **Output:**
  - Transcript text
  - Evaluation metrics and processing time.

### Evaluation Metrics
The system uses standard metrics in ASR tasks:
- **WER (Word Error Rate):** Word error rate.
- **CER (Character Error Rate):** Character error rate.
- **Processing Time:** Processing time from input to output.

### Ablation Study
1. Baseline: *Wav2Vec2 Pretrained*
2. *Wav2Vec2 Fine-tuned*
3. *Wav2Vec2 Fine-tuned + Noise Reduction*
4. *Wav2Vec2 Fine-tuned + Noise Reduction + Data Augmentation*
5. *Wav2Vec2 Fine-tuned + Noise Reduction + Data Augmentation + Language Model Rescoring* (Best Proposed)

---

## 4. Installation & Usage

### System Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Librosa

### Installation Steps

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-repo/Automatic-Speech-Recognition-using-Fine-Tuned-Wav2Vec2.git
   cd wav2vec2-asr
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg (If not already installed):**
   - *Ubuntu:* `sudo apt install ffmpeg`
   - *MacOS:* `brew install ffmpeg`

4. **Run the Demo UI:**
   ```bash
   python app.py
   ```
   *(Access `http://localhost:5000` or `http://127.0.0.1:7860` depending on the UI Framework like Flask/Gradio)*

---

## 5. References
- [Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
