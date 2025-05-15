#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIDI Processing Script

This script takes a MIDI file, extracts the first 8 seconds, splits it into two chunks,
tokenizes them using miditok.REMI, then decodes them back to MIDI.
All files are saved in the data/interim folder.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
from miditok import REMI
from miditoolkit.midi import parser as midi_parser
import miditoolkit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # Force flush to see logs immediately
    force=True
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def extract_first_n_seconds(midi_obj, n_seconds=8):
    """Extract the first n seconds of a MIDI file."""
    # Log tempo changes
    logger.info(f"Tempo changes: {len(midi_obj.tempo_changes)}")
    for i, tc in enumerate(midi_obj.tempo_changes[:3]):  # Log first 3 tempo changes
        logger.info(f"Tempo change {i}: time={tc.time}, tempo={tc.tempo}")
    
    # Get the ticks per beat and tempo
    ticks_per_beat = midi_obj.ticks_per_beat
    tempo_changes = midi_obj.tempo_changes
    
    if not tempo_changes:
        logger.warning("No tempo changes found in MIDI. Using default tempo of 120 BPM")
        tempo = 500000  # Default tempo: 120 BPM (500000 microseconds per beat)
    else:
        tempo = tempo_changes[0].tempo
        # In miditoolkit, tempo is stored in BPM rather than microseconds per beat
        if tempo > 1000:  # Probably microseconds per beat
            logger.info(f"Tempo appears to be in microseconds per beat: {tempo}")
            microseconds_per_beat = tempo
        else:  # Probably BPM
            logger.info(f"Tempo appears to be in BPM: {tempo}")
            microseconds_per_beat = 60_000_000 / tempo  # Convert BPM to microseconds per beat
    
    # Calculate ticks per second (MIDI standard formula)
    beats_per_second = 1_000_000 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second
    
    # Calculate end tick for n seconds
    end_tick = int(n_seconds * ticks_per_second)
    logger.info(f"Limiting to first {n_seconds} seconds = {end_tick} ticks " +
                f"(TPQ: {ticks_per_beat}, tempo: {microseconds_per_beat}μs/beat, {60_000_000/microseconds_per_beat:.1f} BPM)")
    
    # Create a new MIDI object with the same ticks_per_beat
    new_midi = midi_parser.MidiFile(ticks_per_beat=ticks_per_beat)
    
    # Copy tempo changes
    for tempo_change in midi_obj.tempo_changes:
        if tempo_change.time <= end_tick:
            new_midi.tempo_changes.append(tempo_change)
    
    # Copy time signature changes
    for ts_change in midi_obj.time_signature_changes:
        if ts_change.time <= end_tick:
            new_midi.time_signature_changes.append(ts_change)
    
    # Copy key signature changes
    for ks_change in midi_obj.key_signature_changes:
        if ks_change.time <= end_tick:
            new_midi.key_signature_changes.append(ks_change)
    
    # Process each track
    for track in midi_obj.instruments:
        # Create a new instrument with the same properties
        new_track = miditoolkit.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
        new_track.notes = [note for note in track.notes if note.start < end_tick]
        
        # Adjust note end times if they exceed our window
        for note in new_track.notes:
            if note.end > end_tick:
                note.end = end_tick
        
        # If track has notes, add it to the new MIDI
        if new_track.notes:
            new_midi.instruments.append(new_track)
    
    return new_midi

def split_midi_in_half(midi_obj):
    """Split a MIDI file into two equal chunks."""
    # Find the end time (max tick among all notes)
    max_tick = 0
    for track in midi_obj.instruments:
        for note in track.notes:
            max_tick = max(max_tick, note.end)
    
    mid_tick = max_tick // 2
    
    # Create chunk A (first half)
    chunk_a = midi_parser.MidiFile(ticks_per_beat=midi_obj.ticks_per_beat)
    for track in midi_obj.instruments:
        new_track = miditoolkit.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
        new_track.notes = [note for note in track.notes if note.start < mid_tick]
        # Adjust note end times
        for note in new_track.notes:
            if note.end > mid_tick:
                note.end = mid_tick
        if new_track.notes:
            chunk_a.instruments.append(new_track)
    
    # Create chunk B (second half)
    chunk_b = midi_parser.MidiFile(ticks_per_beat=midi_obj.ticks_per_beat)
    for track in midi_obj.instruments:
        new_track = miditoolkit.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
        # Create new notes for chunk B
        new_track.notes = [
            miditoolkit.Note(velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end) 
            for note in track.notes 
            if note.start < max_tick and note.start >= mid_tick
        ]
        # Adjust timestamps to start from 0
        for note in new_track.notes:
            note.start = note.start - mid_tick
            note.end = note.end - mid_tick
        if new_track.notes:
            chunk_b.instruments.append(new_track)
    
    # Copy tempo, time signature, and key signature changes to each chunk
    for chunk, start_tick, end_tick in [(chunk_a, 0, mid_tick), 
                                       (chunk_b, mid_tick, max_tick)]:
        # Copy tempo changes
        for tempo_change in midi_obj.tempo_changes:
            if start_tick <= tempo_change.time < end_tick:
                new_time = tempo_change.time - mid_tick if chunk == chunk_b else tempo_change.time
                tc = miditoolkit.TempoChange(tempo=tempo_change.tempo, time=new_time)
                chunk.tempo_changes.append(tc)
        
        # Copy time signature changes
        for ts_change in midi_obj.time_signature_changes:
            if start_tick <= ts_change.time < end_tick:
                new_time = ts_change.time - mid_tick if chunk == chunk_b else ts_change.time
                tsc = miditoolkit.TimeSignature(
                    numerator=ts_change.numerator, 
                    denominator=ts_change.denominator, 
                    time=new_time
                )
                chunk.time_signature_changes.append(tsc)
        
        # Copy key signature changes
        for ks_change in midi_obj.key_signature_changes:
            if start_tick <= ks_change.time < end_tick:
                new_time = ks_change.time - mid_tick if chunk == chunk_b else ks_change.time
                ksc = miditoolkit.KeySignature(
                    key_number=ks_change.key_number, 
                    time=new_time
                )
                chunk.key_signature_changes.append(ksc)
    
    # If no tempo changes were copied, add default tempo
    for chunk in [chunk_a, chunk_b]:
        if not chunk.tempo_changes:
            if midi_obj.tempo_changes:
                tempo = miditoolkit.TempoChange(tempo=midi_obj.tempo_changes[0].tempo, time=0)
                chunk.tempo_changes.append(tempo)
    
    return chunk_a, chunk_b

def process_midi_file(midi_path, output_dir=None):
    """Process a MIDI file: extract 8 sec, split, tokenize, decode, save."""
    if output_dir is None:
        output_dir = INTERIM_DATA_DIR
    
    ensure_directory_exists(output_dir)
    
    # Load the MIDI file
    logger.info(f"Processing MIDI file: {midi_path}")
    midi_obj = midi_parser.MidiFile(midi_path)
    
    # Log MIDI file details
    note_count = sum(len(track.notes) for track in midi_obj.instruments)
    max_tick = 0
    if midi_obj.instruments:
        for track in midi_obj.instruments:
            for note in track.notes:
                max_tick = max(max_tick, note.end)
    
    logger.info(f"MIDI file details: {len(midi_obj.instruments)} tracks, {note_count} notes, " +
               f"max tick: {max_tick}, ticks per beat: {midi_obj.ticks_per_beat}")
    
    # Extract first 8 seconds
    logger.info("Extracting first 8 seconds")
    first_8sec = extract_first_n_seconds(midi_obj, n_seconds=8)
    
    # Save the 8-second clip
    eight_sec_path = os.path.join(output_dir, "8sec_original.mid")
    first_8sec.dump(eight_sec_path)
    
    # Verify the duration of the extracted clip
    max_tick = 0
    for instrument in first_8sec.instruments:
        for note in instrument.notes:
            max_tick = max(max_tick, note.end)
    
    if first_8sec.tempo_changes:
        tempo = first_8sec.tempo_changes[0].tempo
    else:
        tempo = 500000  # Default 120 BPM
    
    seconds = (max_tick * tempo) / (1_000_000 * first_8sec.ticks_per_beat)
    logger.info(f"Saved 8-second clip to {eight_sec_path} (actual duration: {seconds:.2f} seconds, {max_tick} ticks)")
    
    # Split into chunks A and B
    logger.info("Splitting into chunks A and B")
    chunk_a, chunk_b = split_midi_in_half(first_8sec)
    
    # Save chunks
    chunk_a_path = os.path.join(output_dir, "chunk_a_original.mid")
    chunk_b_path = os.path.join(output_dir, "chunk_b_original.mid")
    chunk_a.dump(chunk_a_path)
    chunk_b.dump(chunk_b_path)
    logger.info(f"Saved chunks to {chunk_a_path} and {chunk_b_path}")
    
    # Initialize REMI tokenizer
    logger.info("Initializing REMI tokenizer")
    tokenizer = REMI()
    
    # Process each chunk
    for name, chunk, path in [
        ("A", chunk_a, chunk_a_path),
        ("B", chunk_b, chunk_b_path)
    ]:
        # Tokenize
        logger.info(f"Tokenizing chunk {name}")
        tokens = tokenizer(chunk)  # Tokenize
        
        # Decode
        logger.info(f"Decoding chunk {name}")
        decoded_midi = tokenizer.decode(tokens)  # Decode
        
        # Save decoded version
        decoded_path = os.path.join(output_dir, f"chunk_{name.lower()}_decoded.mid")
        
        # Handle different MIDI object types (miditoolkit or symusic Score)
        if hasattr(decoded_midi, 'dump'):
            # Old miditoolkit objects
            decoded_midi.dump(decoded_path)
        elif hasattr(decoded_midi, 'dump_midi'):
            # New symusic.core.Score objects
            decoded_midi.dump_midi(decoded_path)
        else:
            # Try other common methods
            if hasattr(decoded_midi, 'write'):
                decoded_midi.write(decoded_path)
            elif hasattr(decoded_midi, 'write_midi'):
                decoded_midi.write_midi(decoded_path)
            elif hasattr(decoded_midi, 'save'):
                decoded_midi.save(decoded_path)
            else:
                logger.error(f"Could not save MIDI: unknown object type {type(decoded_midi)}")
                logger.error(f"Available methods: {dir(decoded_midi)}")
                raise ValueError(f"Could not save MIDI file of type {type(decoded_midi)}")
                        
        logger.info(f"Saved decoded chunk {name} to {decoded_path}")

def main():
    """Main function to parse arguments and process MIDI file."""
    parser = argparse.ArgumentParser(
        description="Process a MIDI file using miditok.REMI tokenizer"
    )
    parser.add_argument(
        "midi_path", type=str, help="Path to the input MIDI file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save output files (default: data/interim)"
    )
    
    args = parser.parse_args()
    
    process_midi_file(args.midi_path, args.output_dir)
    logger.info("Processing complete")

if __name__ == "__main__":
    main()
