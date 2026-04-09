"""
CodeAlpha Task 3: Music Generation with AI
==========================================
LSTM-based music generator using music21 for MIDI processing.
Trains on a small built-in note corpus, then generates new music.

Install dependencies:
    pip install music21 numpy tensorflow

Run:
    python music_gen.py
"""

import os
import random
import numpy as np

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    import music21
    from music21 import stream, note, chord, instrument, tempo
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False
    print("[WARNING] music21 not found. Install: pip install music21")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[WARNING] TensorFlow not found. Install: pip install tensorflow")


# ── Built-in training corpus (note sequences) ─────────────────────────────────
# Each string = a note name or chord (dot-separated notes)
CORPUS = [
    # Classic melodic patterns
    "C4", "E4", "G4", "C5", "G4", "E4", "C4",
    "D4", "F4", "A4", "D5", "A4", "F4", "D4",
    "E4", "G4", "B4", "E5", "B4", "G4", "E4",
    "F4", "A4", "C5", "F5", "C5", "A4", "F4",
    "G4", "B4", "D5", "G5", "D5", "B4", "G4",
    "A4", "C5", "E5", "A5", "E5", "C5", "A4",
    "B4", "D5", "F5", "B5", "F5", "D5", "B4",
    "C4", "E4", "G4", "B4", "G4", "E4", "C4",
    # Minor patterns
    "A3", "C4", "E4", "A4", "E4", "C4", "A3",
    "B3", "D4", "F4", "B4", "F4", "D4", "B3",
    "C4", "Eb4", "G4", "C5", "G4", "Eb4", "C4",
    "D4", "F4", "Ab4", "D5", "Ab4", "F4", "D4",
    # Chords
    "C4.E4.G4", "F4.A4.C5", "G4.B4.D5", "C4.E4.G4",
    "A3.C4.E4", "D4.F4.A4", "E4.G4.B4", "A3.C4.E4",
    # Scale runs
    "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
    "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4",
    "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5",
    # Arpeggios
    "C4", "E4", "G4", "C5", "E5", "G5", "C6",
    "G3", "B3", "D4", "G4", "B4", "D5", "G5",
    # Repeated for more training data
    "C4", "E4", "G4", "C5", "G4", "E4", "C4",
    "D4", "F4", "A4", "D5", "A4", "F4", "D4",
    "E4", "G4", "B4", "E5", "B4", "G4", "E4",
    "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
    "A3", "C4", "E4", "A4", "E4", "C4", "A3",
    "C4.E4.G4", "F4.A4.C5", "G4.B4.D5", "C4.E4.G4",
] * 3  # Repeat corpus 3x for more training data


# ── Data preparation ──────────────────────────────────────────────────────────
def prepare_sequences(notes: list, seq_len: int = 16):
    """Convert note list to integer sequences for LSTM training."""
    vocab = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for n, i in note_to_int.items()}

    X, y = [], []
    for i in range(len(notes) - seq_len):
        X.append([note_to_int[n] for n in notes[i:i + seq_len]])
        y.append(note_to_int[notes[i + seq_len]])

    X = np.array(X)
    y = np.array(y)
    return X, y, note_to_int, int_to_note, len(vocab)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(vocab_size: int, seq_len: int) -> "tf.keras.Model":
    model = Sequential([
        Embedding(vocab_size, 64, input_length=seq_len),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


# ── Generation ────────────────────────────────────────────────────────────────
def generate_notes(model, seed_seq: list, int_to_note: dict,
                   vocab_size: int, n_notes: int = 64, temperature: float = 0.8) -> list:
    """Generate n_notes new notes using temperature sampling."""
    generated = list(seed_seq)
    seq = list(seed_seq)

    for _ in range(n_notes):
        x = np.array([seq])
        preds = model.predict(x, verbose=0)[0].astype("float64")

        # Temperature sampling
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        idx = np.random.choice(len(preds), p=preds)

        generated.append(int_to_note[idx])
        seq = seq[1:] + [idx]

    return generated


# ── MIDI output ───────────────────────────────────────────────────────────────
def notes_to_midi(note_list: list, output_path: str = "generated_music.mid",
                  bpm: int = 120) -> str:
    """Convert generated note strings to a MIDI file."""
    s = stream.Score()
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=bpm))

    for item in note_list:
        if "." in item:
            # Chord
            pitches = item.split(".")
            c = chord.Chord([note.Note(p) for p in pitches])
            c.duration.quarterLength = 0.5
            part.append(c)
        else:
            # Single note
            try:
                n = note.Note(item)
                n.duration.quarterLength = 0.5
                part.append(n)
            except Exception:
                pass

    s.append(part)
    s.write("midi", fp=output_path)
    return output_path


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    SEQ_LEN = 16
    EPOCHS = 50
    BATCH_SIZE = 32
    MODEL_PATH = "music_model.h5"
    OUTPUT_MIDI = "generated_music.mid"

    if not TF_OK:
        print("TensorFlow is required. Run: pip install tensorflow")
        return
    if not MUSIC21_OK:
        print("music21 is required. Run: pip install music21")
        return

    print("=" * 55)
    print("  CodeAlpha Task 3 — AI Music Generator")
    print("=" * 55)

    # 1. Prepare data
    print(f"\n[1/4] Preparing sequences from {len(CORPUS)} notes...")
    X, y, note_to_int, int_to_note, vocab_size = prepare_sequences(CORPUS, SEQ_LEN)
    print(f"      Vocabulary size : {vocab_size}")
    print(f"      Training samples: {len(X)}")

    # 2. Build / load model
    if os.path.exists(MODEL_PATH):
        print(f"\n[2/4] Loading existing model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
    else:
        print("\n[2/4] Building LSTM model...")
        model = build_model(vocab_size, SEQ_LEN)

        print(f"\n[3/4] Training for up to {EPOCHS} epochs...")
        callbacks = [
            ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="loss", verbose=0),
            EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, verbose=1),
        ]
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
        print(f"      Model saved to {MODEL_PATH}")

    # 3. Generate notes
    print("\n[4/4] Generating new music...")
    seed_indices = random.choice(range(len(X)))
    seed_seq = list(X[seed_indices])
    generated = generate_notes(model, seed_seq, int_to_note, vocab_size,
                                n_notes=80, temperature=0.85)

    # 4. Export to MIDI
    midi_path = notes_to_midi(generated, OUTPUT_MIDI)
    print(f"\n✅ Music generated! MIDI saved to: {os.path.abspath(midi_path)}")
    print("   Open the .mid file with VLC, Windows Media Player, or GarageBand.")

    # Try to play with music21's default player
    try:
        sp = music21.environment.Environment()
        sp.write("midiPath", "")
        from music21 import converter
        score = converter.parse(midi_path)
        score.show("midi")
    except Exception:
        print("   (Auto-play failed — open the MIDI file manually.)")


if __name__ == "__main__":
    main()
