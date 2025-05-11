#!/usr/bin/env python
"""
Keyword Spotter – Keras edition 🥑 (auto‑val split)
==================================================
Your project only has **Train/** and **Test/**; no Validation/. This version
creates a 10 % hold‑out slice from Train at run‑time, so you don’t need to
rename folders.

Folder layout it expects now:
```
PROJECTNR2/
 ├─ Train/
 ├─ Test/
 └─ KeyWordDetection.py
```
Inside each, sub‑folders per word plus optional `_silence_`.

Setup / run
-----------
```bash
pip install tensorflow  # TF‑2.x
python KeyWordDetection.py
```
"""
from __future__ import annotations
import os, random, pathlib
import tensorflow as tf

# -------------------------------------------------
# CONFIG 🎛️
# -------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT                    # Train/ / Test/ live here
WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
]
USE_SILENCE_CLASS = False           # True ⇒ "_silence_" is its own label
VAL_FRACTION      = 0.1             # portion of Train used for validation

BATCH_SIZE  = 64
EPOCHS      = 15
LR         = 1e-3
SAMPLE_RATE = 16_000
NUM_MELS    = 40
CKPT_OUT    = "kws_keras.h5"

random.seed(13)

# -------------------------------------------------
# Scan a folder → lists of wav paths & labels
# -------------------------------------------------

def scan_dir(dir_path: pathlib.Path):
    words = [w.lower() for w in WORDS]
    has_sil = USE_SILENCE_CLASS and (dir_path / "_silence_").is_dir()
    unknown_i = len(words)
    silence_i = len(words) if not has_sil else len(words) + 1

    paths, labels = [], []
    for folder in dir_path.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if has_sil and name == "_silence_":
            label = silence_i
        elif name in words:
            label = words.index(name)
        else:
            label = unknown_i
        for wav in folder.glob("*.wav"):
            paths.append(str(wav))
            labels.append(label)

    num_classes = len(WORDS) + 1 + int(has_sil)
    return paths, labels, num_classes

# -------------------------------------------------
# Audio → log‑Mel helper
# -------------------------------------------------
mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MELS,
    num_spectrogram_bins=129,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=0.0,
    upper_edge_hertz=8000.0,
)

def wav_to_logmel(path: tf.Tensor):
    audio_bin = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
    wav = tf.squeeze(wav, -1)
    wav_len = tf.shape(wav)[0]
    wav = tf.cond(
        wav_len < SAMPLE_RATE,
        lambda: tf.pad(wav, [[0, SAMPLE_RATE - wav_len]]),
        lambda: wav[:SAMPLE_RATE],
    )
    stft = tf.signal.stft(wav, frame_length=256, frame_step=128, fft_length=256)
    spec = tf.abs(stft)
    mel = tf.tensordot(spec, mel_weight_matrix, 1)
    mel.set_shape(spec.shape[:-1].concatenate([NUM_MELS]))
    log_mel = tf.math.log(mel + 1e-6)
    return log_mel

# -------------------------------------------------
# Dataset builders
# -------------------------------------------------

def make_tf_dataset(paths, labels, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=13)
    ds = ds.map(lambda p, y: (tf.expand_dims(wav_to_logmel(p), -1), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------------------------------
# Model – tiny CNN
# -------------------------------------------------

def make_model(num_classes, input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    train_root = DATA_ROOT / "Train"
    test_root  = DATA_ROOT / "Test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise SystemExit("Expecting Train/ and Test/ folders next to the script.")

    train_paths, train_labels, n_classes = scan_dir(train_root)
    test_paths,  test_labels,  _        = scan_dir(test_root)

    # === manual train/val split ===
    combined = list(zip(train_paths, train_labels))
    random.shuffle(combined)
    val_size = int(len(combined) * VAL_FRACTION)
    val_combo = combined[:val_size]
    train_combo = combined[val_size:]

    t_paths, t_labels = zip(*train_combo)
    v_paths, v_labels = zip(*val_combo) if val_combo else ([], [])

    train_ds = make_tf_dataset(list(t_paths), list(t_labels), BATCH_SIZE, shuffle=True)
    val_ds   = make_tf_dataset(list(v_paths), list(v_labels), BATCH_SIZE, shuffle=False) if v_paths else None
    test_ds  = make_tf_dataset(test_paths,  test_labels,  BATCH_SIZE, shuffle=False)

    # Snag frame dimension from one batch
    for xs, _ in train_ds.take(1):
        frames = xs.shape[1]
    model = make_model(n_classes, (frames, NUM_MELS, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    callbacks = [tf.keras.callbacks.ModelCheckpoint(CKPT_OUT, save_best_only=True,
                                                    monitor="val_accuracy" if val_ds else "accuracy",
                                                    save_weights_only=False)]

    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

    best_model = tf.keras.models.load_model(CKPT_OUT)
    loss, acc = best_model.evaluate(test_ds, verbose=0)
    print(f"🏁 Test accuracy {acc*100:5.2f}% – model saved → {CKPT_OUT}")


if __name__ == "__main__":
    main()
