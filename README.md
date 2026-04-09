# CodeAlpha_MusicGenerationAI
# CodeAlpha Task 3 — Music Generation with AI

## Features
- LSTM neural network trained on melodic note patterns
- Built-in corpus of 200+ notes (no dataset download needed)
- Temperature-based sampling for creative variation
- Outputs a `.mid` MIDI file playable in any media player
- Auto-saves trained model (`music_model.h5`) — reuses it on next run

## Setup
```bash
pip install -r requirements.txt
python music_gen.py
```

## How It Works
1. **Corpus** → 200+ note/chord sequences (C-major, A-minor, arpeggios)
2. **Preprocessing** → notes tokenized to integers, 16-step sliding windows
3. **Model** → Embedding → LSTM(256) → Dropout → LSTM(256) → Dense(vocab)
4. **Training** → Sparse categorical cross-entropy, Adam optimizer, EarlyStopping
5. **Generation** → Temperature sampling (0.85) from seed sequence → 80 new notes
6. **Export** → `music21` writes a `.mid` MIDI file at 120 BPM

## Output
- `generated_music.mid` — open with VLC, Windows Media Player, GarageBand, or MuseScore
- `music_model.h5` — saved model weights (reused on subsequent runs)

## Use Your Own MIDI Data
Replace `CORPUS` with notes parsed from your own MIDI files:
```python
from music21 import converter, note, chord
score = converter.parse("your_file.mid")
notes = []
for element in score.flat.notes:
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        notes.append(".".join(str(n) for n in element.normalOrder))
```
