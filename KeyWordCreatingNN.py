"""
Keyword Spotter – Keras edition (3 models)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "2"
os.environ["GRPC_VERBOSITY"]       = "ERROR"

import random, pathlib, sys
import tensorflow as tf

# CONFIG 
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT
WORDS = [
    "left", "right", "up", "down", "home", "marvin",
    "cat", "on", "stop", "yes"
]
USE_SILENCE_CLASS = True

BATCH_SIZE  = 256
EPOCHS      = 15
LR          = 1e-3
SAMPLE_RATE = 16_000
NUM_MELS    = 40
MODEL_TYPE  = "medium"  # "tiny", "medium", "heavy"
CKPT_OUT    = "kws_keras_" + MODEL_TYPE +"_Silence_" + ".h5"

random.seed(322)

# Scan directory for labeled .wav files and return paths, labels, and class count
def scan_dir(dir_path: pathlib.Path):
    words = [w.lower() for w in WORDS]  # Normalize known labels (to the lower case)
    has_sil = USE_SILENCE_CLASS and (dir_path / "_silence_").is_dir()  # Check for optional silence class

    unk_i = len(words)  # Index for unknown words
    sil_i = len(words) if not has_sil else len(words) + 1  # Index for silence if it's used

    paths, labels = [], []

    for folder in dir_path.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name.lower()
        # Determine label index
        if has_sil and name == "_silence_":
            label = sil_i
        elif name in words:
            label = words.index(name)
        else:
            label = unk_i  # Unknown class

        # Collect all .wav files in this folder
        for wav in folder.glob("*.wav"):
            paths.append(str(wav))
            labels.append(label)

    # Total number of classes = known words + unknown + optional silence
    n_classes = len(WORDS) + 1 + int(has_sil)
    return paths, labels, n_classes

# AUDIO TO MEL
mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MELS,
    num_spectrogram_bins=129,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=0.0,
    upper_edge_hertz=8000.0,
)


# Convert a .wav file path to a log-mel spectrogram tensor
def wav_to_logmel(path: tf.Tensor):
    audio_bin = tf.io.read_file(path)  # Read audio file
    wav, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)  # Decode to waveform
    wav = tf.squeeze(wav, -1)  # Remove extra channel dim

    wav_len = tf.shape(wav)[0]
    # Pad or crop audio to fixed length (1 second) crop or set with 0,0,0...
    wav = tf.cond(
        wav_len < SAMPLE_RATE,
        lambda: tf.pad(wav, [[0, SAMPLE_RATE - wav_len]]),
        lambda: wav[:SAMPLE_RATE],
    )

    # Compute spectrogram
    stft = tf.signal.stft(wav, frame_length=256, frame_step=128, fft_length=256) # 3.2 + 1.7j, Magnitude, Phase
    # abs(a + bj) = sqrt(a² + b²) converts img numbers to real
    spec = tf.abs(stft)

    # Apply mel filterbank
    mel = tf.tensordot(spec, mel_weight_matrix, 1)
    mel.set_shape(spec.shape[:-1].concatenate([NUM_MELS]))  # Explicit shape for TF graph

    # Log scale for better numerical behavior
    log_mel = tf.math.log(mel + 1e-6)
    return log_mel

# Create a TensorFlow dataset pipeline with optional shuffling and batching
def make_ds(paths, labels, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=322)  # Shuffle for training

    # Convert .wav to log-mel and add channel dim
    ds = ds.map(lambda p, y: (tf.expand_dims(wav_to_logmel(p), -1), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# MODELS
def make_model_tiny(n_classes, frame_dim): # small model
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(frame_dim, NUM_MELS, 1)),
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

def make_model_medium(n_classes, frame_dim): # medium model
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

def make_model_heavy(n_classes, frame_dim): # Big model
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
        tf.keras.layers.Dropout(0.3),
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
    val_ds   = make_ds(val_paths,   val_labels,   BATCH_SIZE, shuffle=True)
    test_ds  = make_ds(test_paths,  test_labels,  BATCH_SIZE, shuffle=True)

    for xs, _ in train_ds.take(1):
        frames = xs.shape[1]

    model = get_model(MODEL_TYPE, n_classes, frames)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(  # create checkpoints and save best models
        CKPT_OUT, save_best_only=True, monitor="val_accuracy", save_weights_only=False)

    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[ckpt_cb])

    best_model = tf.keras.models.load_model(CKPT_OUT)
    loss, acc = best_model.evaluate(test_ds, verbose=0)
    print(f"🏁 Test accuracy {acc*100:5.2f}% – model ({MODEL_TYPE}) saved → {CKPT_OUT}")


main()
