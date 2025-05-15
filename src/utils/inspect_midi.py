#!/usr/bin/env python3
import miditoolkit

# Load the MIDI file
file_path = "data/interim/8sec_original.mid"
midi = miditoolkit.MidiFile(file_path)

# Get file info
print(f"File: {file_path}")
print(f"Ticks per beat: {midi.ticks_per_beat}")

# Get tempo changes
print("\nTempo Changes:")
for i, tc in enumerate(midi.tempo_changes):
    print(f"  {i}: time={tc.time}, tempo={tc.tempo} μs/beat (≈ {60_000_000/tc.tempo:.2f} BPM)")

# Calculate total duration
max_tick = 0
for instrument in midi.instruments:
    for note in instrument.notes:
        max_tick = max(max_tick, note.end)

# If no tempo change, use default 120 BPM
if not midi.tempo_changes:
    print("\nNo tempo information, using default 120 BPM")
    tempo = 500000  # 500,000 μs/beat = 120 BPM
else:
    # Use the first tempo change
    tempo = midi.tempo_changes[0].tempo

# Calculate duration in seconds
seconds = max_tick * tempo / (1000000 * midi.ticks_per_beat)
minutes = seconds / 60
print(f"\nMax tick: {max_tick}")
print(f"Estimated duration: {seconds:.2f} seconds ({int(minutes)}:{int((minutes % 1) * 60):02d} minutes)")

# Inspect time signatures
print("\nTime Signatures:")
for i, ts in enumerate(midi.time_signature_changes):
    print(f"  {i}: time={ts.time}, {ts.numerator}/{ts.denominator}")

# Count notes
note_count = sum(len(instrument.notes) for instrument in midi.instruments)
print(f"\nTotal notes: {note_count}")

# Inspect first few notes of first track
if midi.instruments:
    print(f"\nFirst track ({midi.instruments[0].name}, program={midi.instruments[0].program}):")
    for i, note in enumerate(sorted(midi.instruments[0].notes, key=lambda x: x.start)[:10]):
        print(f"  {i}: start={note.start}, end={note.end}, pitch={note.pitch}, velocity={note.velocity}")
