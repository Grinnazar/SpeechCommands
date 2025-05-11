#!/usr/bin/env python
"""
Live Keyword Listener 🔊→🪄
==========================
Small helper that loads the **kws_keras.h5** model you just trained and listens
on the default microphone. Every second it prints the word it thinks you said
(if confidence > THRESH).

Setup
-----
```bash
pip install tensorflow sounddevice
python live_keyword_listener.py
```
Make sure `kws_keras.h5` is in the same folder (or tweak `MODEL_PATH`).

Config knobs are at the top: `WORDS`, `USE_SILENCE_CLASS`, and detection
`THRESH`.
"""
from __future__ import annotations
import os, pathlib, time
import numpy as np
import tensorflow as tf
import sounddevice as sd

# -------------------- CONFIG --------------------
MODEL_PATH = "kws_keras.h5"           # ← path to the saved Keras model
WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
]
USE_SILENCE_CLASS = False             # True if you trained with "_silence_" as its own label
THRESH = 0.75                         # confidence threshold to print the word

SAMPLE_RATE = 16_000                  # must match training
NUM_MELS    = 40

# ------------------------------------------------
# Load model & prep label lookup
# ------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
labels = WORDS + ["unknown"] + (["silence"] if USE_SILENCE_CLASS else [])

# ------------------------------------------------
# Audio → log‑Mel helper (numpy version)
# ------------------------------------------------
mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MELS,
    num_spectrogram_bins=129,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=0.0,
    upper_edge_hertz=8000.0,
).numpy()

def wav_to_logmel_np(wav: np.ndarray) -> np.ndarray:
    """wav (N,) float32 in [-1, 1] → log‑Mel (frames, 40, 1)"""
    if wav.shape[0] < SAMPLE_RATE:
        wav = np.pad(wav, (0, SAMPLE_RATE - wav.shape[0]))
    else:
        wav = wav[:SAMPLE_RATE]

    # STFT
    stft = np.abs(tf.signal.stft(wav, frame_length=256, frame_step=128, fft_length=256).numpy())
    mel = np.dot(stft, mel_weight_matrix)
    log_mel = np.log(mel + 1e-6)
    log_mel = log_mel[..., np.newaxis]  # (frames, 40, 1)
    return log_mel.astype(np.float32)

# ------------------------------------------------
# Main loop – record 1 s chunks, run inference
# ------------------------------------------------
print("🎙️  Listening… Speak a keyword (Ctrl‑C to quit)")
try:
    while True:
        audio = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        wav_np = audio[:, 0]  # shape (16000,)
        log_mel = wav_to_logmel_np(wav_np)
        log_mel = np.expand_dims(log_mel, axis=0)  # batch dim
        preds = model.predict(log_mel, verbose=0)[0]
        top_i = int(np.argmax(preds))
        conf  = float(preds[top_i])
        if conf >= THRESH:
            print(f"Heard: {labels[top_i]}  (p={conf:.2f})")
        else:
            print("… (no clear keyword)")
        time.sleep(0.1)  # tiny pause so prints don’t spam
except KeyboardInterrupt:
    print("👋 Exiting")
