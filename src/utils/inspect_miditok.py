#!/usr/bin/env python3
import os
from miditok import REMI
from miditoolkit.midi import parser as midi_parser
import miditoolkit

# Load a MIDI file
midi_path = "data/processed/ddPn08-maestro-v3.0.0/0.midi"
midi_obj = midi_parser.MidiFile(midi_path)

# Initialize REMI tokenizer
tokenizer = REMI()

# Tokenize
tokens = tokenizer(midi_obj)
print(f"Tokens type: {type(tokens)}")

# Decode
decoded = tokenizer.decode(tokens)
print(f"Decoded type: {type(decoded)}")
print(f"Decoded attributes: {dir(decoded)}")

# Try various methods
try:
    import sys
    output_path = os.path.join("data/interim", "test_output.mid")
    print(f"Trying to save to {output_path}")
    
    # Try method 1: write
    if hasattr(decoded, 'write'):
        print("Using write method")
        decoded.write(output_path)
    # Try method 2: save
    elif hasattr(decoded, 'save'):
        print("Using save method")
        decoded.save(output_path)
    # Try method 3: to_file
    elif hasattr(decoded, 'to_file'):
        print("Using to_file method")
        decoded.to_file(output_path) 
    else:
        print("No standard save method found, trying conversion...")
        # Try to convert back to miditoolkit
        try:
            print("Available symusic modules:", dir(sys.modules.get('symusic', {})))
            if 'symusic' in sys.modules:
                import symusic
                if hasattr(symusic, 'to_miditoolkit'):
                    print("Converting with symusic.to_miditoolkit")
                    miditool_obj = symusic.to_miditoolkit(decoded)
                    miditool_obj.dump(output_path)
                else:
                    print("No conversion method found in symusic")
            else:
                print("Symusic module not found")
            
            print("Attempting manual conversion...")
            # Print the structure to find conversion paths
            for attr in dir(decoded):
                print(f"- {attr}")
                try:
                    value = getattr(decoded, attr)
                    if callable(value):
                        print(f"  (callable) {type(value)}")
                    else:
                        print(f"  {type(value)}")
                except Exception as e:
                    print(f"  Error: {e}")
        except Exception as e:
            print(f"Conversion error: {e}")

except Exception as e:
    print(f"Error: {e}")
