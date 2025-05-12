#!/usr/bin/env python
"""
Keyword Spotter – Keras edition (трійця моделей)
"""

from __future__ import annotations
import random, pathlib, sys
import tensorflow as tf

# === CONFIG ===
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT
WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
]
USE_SILENCE_CLASS = True

BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 1e-3
SAMPLE_RATE = 16_000
NUM_MELS    = 40
CKPT_OUT    = "kws_keras.h5"
MODEL_TYPE  = "medium"  # "tiny", "medium", "heavy"

random.seed(13)

# === DATA LOADER ===
def scan_dir(dir_path: pathlib.Path):
    words = [w.lower() for w in WORDS]
    has_sil = USE_SILENCE_CLASS and (dir_path / "_silence_").is_dir()
    unk_i   = len(words)
    sil_i   = len(words) if not has_sil else len(words) + 1

    paths, labels = [], []
    for folder in dir_path.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if has_sil and name == "_silence_":
            label = sil_i
        elif name in words:
            label = words.index(name)
        else:
            label = unk_i
        for wav in folder.glob("*.wav"):
            paths.append(str(wav))
            labels.append(label)

    n_classes = len(WORDS) + 1 + int(has_sil)
    return paths, labels, n_classes

# === AUDIO TO MEL ===
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
    mel  = tf.tensordot(spec, mel_weight_matrix, 1)
    mel.set_shape(spec.shape[:-1].concatenate([NUM_MELS]))
    log_mel = tf.math.log(mel + 1e-6)
    return log_mel

# === DATASET CREATOR ===
def make_ds(paths, labels, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=13)
    ds = ds.map(lambda p, y: (tf.expand_dims(wav_to_logmel(p), -1), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# === MODELS ===
def make_model_tiny(n_classes, frame_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(frame_dim, NUM_MELS, 1)),
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

def make_model_medium(n_classes, frame_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(frame_dim, NUM_MELS, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

def make_model_heavy(n_classes, frame_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(frame_dim, NUM_MELS, 1)),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

def get_model(type: str, n_classes: int, frame_dim: int):
    if type == "tiny":
        return make_model_tiny(n_classes, frame_dim)
    elif type == "medium":
        return make_model_medium(n_classes, frame_dim)
    elif type == "heavy":
        return make_model_heavy(n_classes, frame_dim)
    else:
        raise ValueError(f"Invalid model type: {type}")

# === MAIN ===
def main():
    train_root = DATA_ROOT / "Train"
    valid_root = DATA_ROOT / "Valid"
    test_root  = DATA_ROOT / "Test"

    missing = [p for p in (train_root, valid_root, test_root) if not p.is_dir()]
    if missing:
        print("Missing folders:", *missing, sep="\n - ")
        sys.exit(1)

    train_paths, train_labels, n_classes = scan_dir(train_root)
    val_paths,   val_labels,   _        = scan_dir(valid_root)
    test_paths,  test_labels,  _        = scan_dir(test_root)

    train_ds = make_ds(train_paths, train_labels, BATCH_SIZE, shuffle=True)
    val_ds   = make_ds(val_paths,   val_labels,   BATCH_SIZE, shuffle=False)
    test_ds  = make_ds(test_paths,  test_labels,  BATCH_SIZE, shuffle=False)

    for xs, _ in train_ds.take(1):
        frames = xs.shape[1]

    model = get_model(MODEL_TYPE, n_classes, frames)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        CKPT_OUT, save_best_only=True, monitor="val_accuracy", save_weights_only=False)

    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[ckpt_cb])

    best_model = tf.keras.models.load_model(CKPT_OUT)
    loss, acc = best_model.evaluate(test_ds, verbose=0)
    print(f"🏁 Test accuracy {acc*100:5.2f}% – модель ({MODEL_TYPE}) збережено → {CKPT_OUT}")

if __name__ == "__main__":
    main()
