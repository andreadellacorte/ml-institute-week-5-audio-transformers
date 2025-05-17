from miditok import REMI, TokenizerConfig

def create_remi_tokenizer():
    """
    Create and return a REMI tokenizer instance.
    The tokenizer is configured to match the one used during data preprocessing.

    Returns:
        REMI tokenizer instance
    """
    # Create a TokenizerConfig with the desired settings
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=False,
        use_sustain_pedals=True,
        use_rests=True,
        use_tempos=True,
        use_dynamics=True,
        use_time_signatures=True,
    )

    # Initialize the REMI tokenizer with the configuration
    tokenizer = REMI(config)
    return tokenizer
