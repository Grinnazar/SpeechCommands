#!/usr/bin/env python3
"""
🎙️  Live Keyword Listener
---------------------------------------------------------------------

Voice triggers:

  on   → cat      • opens a random cat GIF 
  stop → yes      • kills the focused window
  left / right    • switch virtual desktops
  up   / down     • Alt-Tab forward / backward
  home            • show desktop
  marvin          • sarcastic Marvin notification
"""

import os, time, subprocess, collections
import numpy as np
import sounddevice as sd
import tensorflow as tf

# ENV NOISE-CANCELLING
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "2"
os.environ["GRPC_VERBOSITY"]       = "ERROR"

# CONFIG
MODEL_PATH = "kws_keras_medium_Silence_ALL.h5"

WORDS = [
    "left", "right", "up", "down", "home", "marvin",
    "cat", "on", "stop", "yes"
]

USE_SILENCE      = True             # set False if your model lacks silence
THRESH           = 0.75             # confidence to accept a word
SEQ_TIMEOUT      = 1.2              # max gap between combo words (s)
REPEAT_TIMEOUT   = 2.0              # ignore same word inside this window
SAMPLE_RATE      = 16_000
NUM_MELS         = 40
WINDOW           = SAMPLE_RATE      # 1-second log-Mel window
HOP              = int(0.10 * SAMPLE_RATE)   # 100 ms hops

# MODEL + FEATURE EXTRACTOR
model  = tf.keras.models.load_model(MODEL_PATH)
labels = WORDS + ["unknown"] + (["silence"] if USE_SILENCE else [])

mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MELS, num_spectrogram_bins=129,
    sample_rate=SAMPLE_RATE, lower_edge_hertz=0.0, upper_edge_hertz=8000.0
).numpy()

def wav_to_logmel_np(wav: np.ndarray) -> np.ndarray:
    """1-D float32 wav → (frames, 40, 1) log-Mel"""
    stft = np.abs(tf.signal.stft(wav, 256, 128, 256).numpy())
    mel  = np.dot(stft, mel_weight_matrix)
    log  = np.log(mel + 1e-6)[..., np.newaxis]
    return log.astype(np.float32)

# HELPERS LINUX FUNCTIONS
def run(*cmd):
    subprocess.Popen(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def open_cat_gif():        run("xdg-open", "https://cataas.com/cat/gif")
def close_active_window(): run("xdotool", "getwindowfocus", "windowkill")

def switch_workspace(d):
    cmd = ["xdotool", "set_desktop", "--relative"]
    if d < 0:
        cmd.append("--")          # stop option-parsing before "-1"
    cmd.append(str(d))
    run(*cmd)

def alt_tab(d):            run("xdotool", "key",
                               "Alt_L+Tab" if d > 0 else "Alt_L+Shift_L+Tab")
def show_desktop():        run("wmctrl", "-k", "on")
def marvin_notify():       run("notify-send", "Marvin",
                               "Here I am, brain the size of a planet…")

# COMBO TABLE (easier to read & extend)
COMBOS = {
    ("on",   "cat"): open_cat_gif,
    ("stop", "yes"): close_active_window,
}

# AUDIO PIPELINE
ring = collections.deque(maxlen=WINDOW)
def audio_cb(indata, frames, time_info, status):
    ring.extend(indata[:, 0])

print("🎙️  Listening… speak away (Ctrl-C to quit)")

# state
last_word, last_time = None, 0.0
last_seen = {}      # word → last trigger time (anti-spam)

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                    blocksize=HOP, callback=audio_cb):
    try:
        while True:
            if len(ring) < WINDOW:         # buffer not yet full
                time.sleep(0.01)
                continue

            wav_np = np.array(ring, dtype=np.float32)
            logmel = np.expand_dims(wav_to_logmel_np(wav_np), 0)
            preds  = model.predict(logmel, verbose=0)[0]
            top_i  = int(np.argmax(preds))
            conf   = float(preds[top_i])
            now    = time.time()

            if conf < THRESH:
                continue

            word = labels[top_i]

            if word in ("silence", "unknown"):
                continue

            # anti-spam: ignore rapid repeats of the *same* word
            if word in last_seen and now - last_seen[word] < REPEAT_TIMEOUT:
                continue
            last_seen[word] = now

            print(f"Heard: {word:<6} (p={conf:.2f})")

            # single-word actions
            if   word == "left":   switch_workspace(-1)
            elif word == "right":  switch_workspace(+1)
            elif word == "up":     alt_tab(+1)
            elif word == "down":   alt_tab(-1)
            elif word == "home":   show_desktop()
            elif word == "marvin": marvin_notify()

            # two-word combos
            if last_word and now - last_time <= SEQ_TIMEOUT:
                action = COMBOS.get((last_word, word))
                if action:
                    action()
                    # reset combo tracker so a third word doesn’t fire again
                    last_word, last_time = None, 0.0
                    continue   # next audio part

            # keep the latest word for the next round
            last_word, last_time = word, now

    except KeyboardInterrupt:
        print("\n👋  Bye")
