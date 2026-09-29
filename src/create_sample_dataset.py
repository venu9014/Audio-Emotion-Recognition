"""
EMOTIVA — Realistic Acoustic Speech & Dataset Generator
Synthesizes distinct acoustic signals for all 8 emotions using source-filter
acoustics (pitch contours, formant filtering, rhythmic prosody, harmonics).

Creates:
  1. 8 standalone demo audio files in `samples/demo_{emotion}.wav`
  2. Multi-actor RAVDESS dataset in `dataset/RAVDESS/` with odd (male) and even (female) actors
"""
import os
import sys
import numpy as np
import scipy.signal as signal
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def synthesize_emotion_speech(
    emotion_code: int,
    actor_id: int = 1,
    rep: int = 1,
    duration: float = 3.0,
    sr: int = 22050
) -> np.ndarray:
    """
    Synthesize acoustic speech signal for an emotion (1..8) with actor-specific pitch.
    Odd actor IDs = male pitch range; Even actor IDs = female pitch range.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    is_female = (actor_id % 2 == 0)

    # Actor baseline pitch: male ~135 Hz, female ~210 Hz with subtle actor offsets
    base_f0 = 210.0 if is_female else 135.0
    base_f0 *= (1.0 + 0.04 * (actor_id % 4 - 1.5))

    # Profiles: (pitch_multiplier, formant_center, syllable_rate, hf_tilt, contour_type)
    # 1: neutral, 2: calm, 3: happy, 4: sad, 5: angry, 6: fearful, 7: disgust, 8: surprised
    profiles = {
        1: (1.0,  500,  2.2, 0.2, 'flat'),     # neutral: steady baseline
        2: (0.9,  420,  1.8, 0.08, 'gentle'),  # calm: relaxed, gentle, smooth cadence
        3: (1.4,  750,  3.8, 0.7, 'bouncy'),   # happy: bright, bouncy, animated
        4: (0.75, 340,  1.3, 0.02, 'down'),    # sad: low, slow, downward slope
        5: (1.7,  850,  4.2, 1.0, 'harsh'),    # angry: loud, high pitch, overdrive
        6: (1.5,  620,  3.9, 0.4, 'tremor'),   # fearful: trembling, rapid flutter
        7: (0.65, 280,  1.6, 0.15, 'creaky'),  # disgust: low creaky vocal fry
        8: (1.3,  780,  3.0, 0.8, 'swoop')     # surprised: dramatic upward swoop
    }

    p_mult, F_c, syl, hf, ctype = profiles.get(emotion_code, (1.0, 500, 2.2, 0.2, 'flat'))
    f0 = base_f0 * p_mult

    # 1. Fundamental frequency contour (Intonation trajectory)
    if ctype == 'down':
        f0_t = f0 - 32.0 * (t / duration) + 2.0 * np.sin(2 * np.pi * 1.5 * t)
    elif ctype == 'swoop':
        f0_t = f0 - 20.0 + 120.0 / (1.0 + np.exp(-12.0 * (t / duration - 0.38)))
    elif ctype == 'tremor':
        f0_t = f0 + 22.0 * np.sin(2 * np.pi * 8.0 * t) + 8.0 * np.sin(2 * np.pi * 15.0 * t)
    elif ctype == 'bouncy':
        f0_t = f0 + 35.0 * np.sin(2 * np.pi * 4.0 * t) + 12.0 * np.sin(2 * np.pi * 8.0 * t)
    elif ctype == 'harsh':
        f0_t = f0 + 42.0 * np.sin(2 * np.pi * 5.0 * t) + 20.0 * np.cos(2 * np.pi * 9.0 * t)
    elif ctype == 'gentle':
        f0_t = f0 + 2.0 * np.sin(2 * np.pi * 1.0 * t)
    elif ctype == 'creaky':
        f0_t = f0 + 5.0 * np.sin(2 * np.pi * 1.8 * t) - 8.0 * (t / duration)
    else:  # flat / neutral
        f0_t = f0 + 4.0 * np.sin(2 * np.pi * 2.0 * t)

    f0_t = np.clip(f0_t, 50.0, 700.0)
    phase = 2 * np.pi * np.cumsum(f0_t) / sr

    # 2. Glottal source pulse + harmonics
    pulse = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    if ctype == 'creaky':
        pulse += 0.9 * np.sin(0.5 * phase) + 0.4 * np.sin(0.25 * phase)  # deep subharmonics for fry

    # 3. Formant bandpass resonator filter
    flow = max(100.0, F_c * 0.6)
    fhigh = min(sr / 2.0 - 100.0, F_c * 1.5)
    b, a = signal.butter(2, [flow / (sr / 2.0), fhigh / (sr / 2.0)], btype='band')
    filtered = signal.lfilter(b, a, pulse)

    # 4. Syllabic prosodic cadence
    env = 0.5 * (1.0 + np.sin(2 * np.pi * syl * t - np.pi / 2))
    env = np.clip(env, 0.12, 1.0)

    # 5. Global phrasing contour
    if ctype == 'down':
        phrase = np.clip(1.0 - 0.5 * (t / duration), 0.2, 1.0)
    elif ctype == 'swoop':
        phrase = np.clip((t / duration) ** 0.8, 0.2, 1.0)
    elif ctype == 'harsh':
        phrase = np.ones_like(t) * 0.95
    elif ctype == 'gentle':
        phrase = 0.6 * np.sin(np.pi * t / duration) ** 0.7
    else:
        phrase = np.sin(np.pi * t / duration) ** 0.5

    sig = filtered * env * phrase

    # Non-linear vocal saturation for anger
    if ctype == 'harsh':
        sig = np.tanh(2.0 * sig)

    # Peak normalization
    max_amp = np.max(np.abs(sig))
    if max_amp > 0:
        sig = 0.92 * sig / max_amp

    return sig.astype(np.float32)


def generate_all_demo_voices(output_dir: str = None):
    """Generate all 8 distinct emotion demo voices in samples/."""
    if output_dir is None:
        output_dir = os.path.join(config.PROJECT_ROOT, "samples")
    os.makedirs(output_dir, exist_ok=True)

    sr = config.SR
    print(f"Generating 8 emotion demo voices in {output_dir}...")

    for code, emo_name in config.EMOTION_LABELS.items():
        signal_out = synthesize_emotion_speech(code, actor_id=1, rep=1, duration=3.2, sr=sr)
        
        # WAV
        filename_wav = f"demo_{emo_name}.wav"
        sf.write(os.path.join(output_dir, filename_wav), signal_out, sr)
        
        # MP3
        filename_mp3 = f"demo_{emo_name}.mp3"
        sf.write(os.path.join(output_dir, filename_mp3), signal_out, sr, format='MP3')
        
        print(f"  [OK] {emo_name.capitalize():<10} -> {filename_wav} & {filename_mp3}")

        # Maintain legacy aliases (sample_neutral.wav/.mp3, sample_happy.wav/.mp3, sample_angry.wav/.mp3)
        if emo_name in ['neutral', 'happy', 'angry']:
            sf.write(os.path.join(output_dir, f"sample_{emo_name}.wav"), signal_out, sr)
            sf.write(os.path.join(output_dir, f"sample_{emo_name}.mp3"), signal_out, sr, format='MP3')

    # Generate multi-emotion long sample (6.5s) for timeline: calm (3.2s) + happy (3.3s)
    s_calm = synthesize_emotion_speech(2, actor_id=1, rep=1, duration=3.2, sr=sr)
    s_happy = synthesize_emotion_speech(3, actor_id=1, rep=1, duration=3.3, sr=sr)
    s_long = np.concatenate([s_calm, s_happy])
    sf.write(os.path.join(output_dir, "sample_long_speech.wav"), s_long, sr)
    sf.write(os.path.join(output_dir, "sample_long_speech.mp3"), s_long, sr, format='MP3')
    print("  [OK] Long Speech (6.5s) -> sample_long_speech.wav & sample_long_speech.mp3")

    # Short audio (< 1s) for boundary tests
    s_short = synthesize_emotion_speech(1, actor_id=1, rep=1, duration=0.3, sr=sr)
    sf.write(os.path.join(output_dir, "sample_too_short.wav"), s_short, sr)
    sf.write(os.path.join(output_dir, "sample_too_short.mp3"), s_short, sr, format='MP3')

    # Corrupted audio for invalid header tests
    with open(os.path.join(output_dir, "sample_corrupted.wav"), "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEcorrupted_garbage_bytes")
    with open(os.path.join(output_dir, "sample_corrupted.mp3"), "wb") as f:
        f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00corrupted_mp3_garbage_data")


def generate_full_ravdess_corpus(base_dir: str = None, num_actors: int = 8, reps_per_emotion: int = 4):
    """
    Generate balanced multi-actor corpus (8 actors, 8 emotions, 4 reps = 256 files).
    Odd actors = male, even actors = female.
    """
    if base_dir is None:
        base_dir = config.DATASET_PATH
    os.makedirs(base_dir, exist_ok=True)
    sr = config.SR
    total_files = 0

    print(f"\nGenerating balanced RAVDESS dataset with {num_actors} actors at {base_dir}...")

    for actor in range(1, num_actors + 1):
        actor_dir = os.path.join(base_dir, f"Actor_{actor:02d}")
        os.makedirs(actor_dir, exist_ok=True)

        for emotion in range(1, 9):
            for rep in range(1, reps_per_emotion + 1):
                filename = f"03-01-{emotion:02d}-01-01-{rep:02d}-{actor:02d}.wav"
                filepath = os.path.join(actor_dir, filename)

                dur = 3.0 + 0.05 * (rep % 3)
                tone = synthesize_emotion_speech(emotion, actor_id=actor, rep=rep, duration=dur, sr=sr)
                sf.write(filepath, tone, sr)
                total_files += 1

    print(f"Generated {total_files} audio files across {num_actors} actors.")


if __name__ == '__main__':
    generate_all_demo_voices()
    generate_full_ravdess_corpus(num_actors=8, reps_per_emotion=4)
