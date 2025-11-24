# Breaking-Speech-to-Text-Models-Using-Adversarial-Injection-Defense
### CYB 590 – Project 3 • Duke University • Fall 2025

---

## 📝 Overview

This project demonstrates how **adversarial audio perturbations**—specifically **high-frequency near-ultrasonic tones** and **Gaussian white noise**—can significantly degrade the performance of modern **speech-to-text (STT)** models such as **OpenAI Whisper** and **Vosk**.

We also implement a **defense pipeline** combining:

- **Low-pass filtering** (6kHz, 8kHz, 12kHz cutoff options)
- **Optional denoising**
- **Spectral artifact reduction**

We evaluate the **effectiveness of both attacks and defenses** using **Word Error Rate (WER)**, waveform analysis, spectrograms, and comparative performance metrics.  
This project follows responsible AI security research practices and ethical disclosure guidelines.

---

## 👥 Project Team & Responsibilities

### **Sakthi Vinayak — Attack Module**
- Designed and implemented adversarial audio attacks  
- High-frequency injection (12–19 kHz tones)  
- White noise injection across full spectrum  
- Generated adversarial datasets, spectrograms, waveform plots  
- Developed comparative WER evaluation script  

### **Nathan Chen — Defense Module**
- Implemented low-pass filter defenses  
- Constructed denoising pipeline  
- Tuned defense parameters (6kHz/8kHz/12kHz cutoffs)  
- Generated defended audio for evaluation  

### **Omkar Sreekanth — Evaluation & STT Pipelines**
- Implemented Whisper and Vosk transcription pipelines  
- Computed WER using jiwer  
- Ran clean vs adversarial vs defended evaluation  
- Produced tables, graphs, and defense–attack tradeoff analysis  

---

## 🎯 Project Goal 
Show how a **high-frequency audio tone** can break STT systems and how **low-pass filtering + denoising** can defend against it.  
We:
1. Built the attack module 
2. Built the defense module
3. Evaluated how the attack generalizes across different STT models  
4. Quantified effects using **Word Error Rate (WER)**

---

## 📦 Repository Structure

- **attack_modules/**
  - 🎧 `adversarial_audio_highfreq_12_19kHz_compressed/`
  - 🎧 `adversarial_audio_whitenoise_compressed/`
  - 📈 `plots_highfreq/`
  - 📈 `plots_whitenoise/`
  - 🎤 `clean_audio/`
  - 🧪 `final_audio_attacks.py`

- **defense/**
  - 🔊 `low_pass_filter.py`
  - 🔈 `denoising.py`
  - 📁 `defended_audio_samples/`
    - 📈 `plots_defense/`

- **evaluation/**
  - 🧠 `whisper_pipeline.py`
  - 🧠 `vosk_pipeline.py`
  - 📊 `wer_evaluation.py`
  - 📄 `combined_results.csv`

- ⚙️ `requirements.txt`   
- 📘 `README.md`

---

## 🧪 Attack Techniques

### **1. High-Frequency Adversarial Injection (12–19 kHz)**

- Near-inaudible sine waves  
- Frequencies: **12k, 15k, 17k, 19k Hz**  
- Amplitudes: **0.01, 0.05, 0.10**  
- Goal: confuse STT feature extraction without being easily perceptible  

### **2. White Noise Injection**

- Gaussian noise with standard deviation:
  - `0.005`, `0.01`, `0.02`, `0.05`
- Attacks entire frequency spectrum  
- Goal: simulate real-world noisy environments that degrade STT accuracy  

### **3. Attack Comparison**

The script outputs:

- Clean transcript  
- HF-attack transcripts + WER  
- White-noise transcripts + WER  
- Side-by-side comparison bar chart  
- Attack ranking (most effective attack type)  

---

## 🛡️ Defense Techniques

### **1. Low-Pass Filtering (LPF)**

Cutoff frequencies tested:

- **6 kHz (strong defense)**  
- **8 kHz (medium)**  
- **12 kHz (mild/slight defense)**  

Filters implemented using **scipy.signal.butter**.

### **2. Optional Denoising**

- Spectral gating  
- Noise-floor attenuation  
- High-frequency suppression  

Evaluated for trade-offs between **speech quality** and **WER recovery**.

---

## 📊 Evaluation Pipeline

### **Speech-to-Text Models Tested**

| Model | Purpose |
|-------|---------|
| **Whisper (base)** | Main evaluation target |
| **Vosk** | Generalization testing |

### **Metrics**

- **WER (Word Error Rate)** using *jiwer*  
- Attack success: `WER(adversarial) − WER(clean)`  
- Defense gain: `WER(defended) − WER(adversarial)`  

### **Outputs**

- Clean vs Adversarial vs Defended WER table  
- Frequency × Amplitude × WER table  
- Defense cutoff × WER table  
- Waveforms and spectrograms  
- Combined attack comparison graph  

---

## 📁 Data Requirements

You may supply:

- Your own audio samples  
- Or any public speech dataset (LibriSpeech, Mozilla Common Voice)  

---

## 🧰 Installation

### Install dependencies:

pip install -r requirements.txt

### Requirements include:
whisper
jiwer
numpy
scipy
librosa
soundfile
matplotlib
pydub
vosk

---

## ⚙️ Setup & Installation

This project runs on:

- Python **3.9+**
- Google Colab **(recommended)**
- Ubuntu 22.04 / macOS 14
- Whisper CPU/GPU inference

---

### 1️⃣ Clone the Repository

git clone https://github.com/cjx12036/Breaking-Speech-to-Text-Models-Using-Adversarial-Injection-Defense.git
cd Breaking-Speech-to-Text-Models-Using-Adversarial-Injection-Defense

### 2️⃣ Create a Virtual Environment (Recommended)
python3 -m venv venv
source venv/bin/activate   # Mac/Linux

#### Windows:
venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Prepare Audio Files

Place your clean audio in:
attack_modules/clean_audio/

Supported formats:
.wav, .flac, .mp3, .ogg

### 5️⃣ Run Adversarial Attack Generation (Sakthi • Attack Module)
python attack_modules/final_audio_attacks.py

Outputs appear under:
attack_modules/adversarial_audio_highfreq_12_19kHz_compressed/
attack_modules/adversarial_audio_whitenoise_compressed/

### 6️⃣ Run Low-Pass Filter Defense (Nathan • Defense Module)
python defense/low_pass_filter.py

Optional denoising:
python defense/denoising.py

Defended audio will be saved under:
defense/defended_audio_samples/
defense/plots_defense/

### 7️⃣ Evaluate STT Performance (Omkar • Evaluation Module)



---

# 🔐 Security Research Best Practices

This project complies with academic responsible-use policies and standard security research ethics.

### ✔ Only evaluate models + audio that YOU own
We **never** attacked:

- live systems  
- proprietary APIs  
- third-party services  

All experiments occurred locally on open-source STT models.

### ✔ Techniques are dual-use — consider misuse risks
High-frequency attacks could be used maliciously to:

- evade transcription  
- disrupt voice assistants  
- inject inaudible commands  

Therefore:

- We release **only benign examples**  
- No harmful payloads  
- Attack code is documented for **academic defense research only**

### ✔ Responsible Disclosure Policy
If we discovered an exploitable vulnerability affecting production systems, we would:

1. Contact the vendor’s security team  
2. Provide minimal reproducible examples  
3. Allow time for patching before public release  

No such disclosure was necessary in this project.

### ✔ Focus on defensive and safety applications
Our primary objective is:

- Helping improve model robustness  
- Advancing adversarial defense research  
- Studying model generalization to real-world noise  

---

# 📈 Results (Summary)

> **High-frequency attacks were more stealthy**  
> **White noise attacks were more destructive to WER**  
> **Low-pass filtering at 6–8 kHz mitigated most HF attacks**  
> **Generalization varied across Whisper vs Vosk**  

Final plots and results are available inside:

`attack_modules/plots_highfreq/`  
`attack_modules/plots_whitenoise/`  
`evaluation/combined_results.csv`

---

# 🎧 Real-World Implications

- Smart speakers and voice assistants can be influenced by inaudible signals  
- Adversaries can hide commands in near-ultrasound frequencies  
- White noise can be used to confuse automated transcription systems  
- Filtering defenses introduce trade-offs between intelligibility and security  

---

## 🤖 Use of AI Systems in This Project

We used AI-based development tools to support coding, debugging, and documentation generation. All design decisions, algorithms, and implementation logic were created and validated by the project team.

### Tools Used
- **ChatGPT, Claude**  
  Used for:
  - generating boilerplate code for signal processing (e.g., waveform plotting, spectrogram setup)
  - debugging Python/NumPy/Librosa issues
  - documentation writing and README formatting
  - code refactoring for cleanliness and reproducibility

### Human Oversight
All generated or suggested code and documentation were:
- manually verified
- edited to ensure correctness
- tested within the project environment
- integrated only after thorough review by the team

We maintain full responsibility for the design, logic, implementation, evaluation, and ethical use of the project.

---

# 📝 Citation


Vinayak, Sakthi; Chen, Nathan; Sreekanth, Omkar.
"Breaking Speech-to-Text Models Using Adversarial Audio Injection & Defense."
Duke University, CYB 590, Fall 2025.

---

