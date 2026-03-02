import numpy as np
import librosa
import soundfile as sf
from PIL import Image
from pathlib import Path
import os

from song_recommender.paths import *

def png_to_audio(png_path : Path,
                 track_id : str, 
                 output_name : str,
                 config,
                 audio_output_base : Path = Path('temp/audio')
                 ) -> None:
    
    audio_output_base = audio_output_base / track_id
    audio_output_path = f'{audio_output_base}/{output_name}.wav'

    # create temporary subdirectory in current to store generated audio
    if not os.path.exists(audio_output_base):
            os.makedirs(audio_output_base)
            print(f'Created subdirectory {audio_output_base} in current directory to save generated audio')

    """
    Best-effort PNG (log-mel *visualization*) -> audio.

    Important:
    - If the PNG is a colormap-rendered plot (RGBA/RGB), this is NOT invertible in general.
      This function still returns audio, but pitch/harmonics may be wrong.
    - Timing usually comes out better than pitch.
    """

    # --- load image ---
    img = Image.open(png_path)
    A = np.asarray(img)

    # crop if you have axes/ticks/borders
    if config['reconstruction']['crop'] is not None:
        top, bottom, left, right = config['reconstruction']['crop']
        A = A[top:A.shape[0]-bottom, left:A.shape[1]-right]

    # ensure 0..1 float
    if A.dtype == np.uint8:
        Af = A.astype(np.float32) / 255.0
    else:
        Af = A.astype(np.float32)
        # if floats are not in [0,1], clip
        Af = np.clip(Af, 0.0, 1.0)

    # --- collapse to a 2D intensity map in [0,1] ---
    # If grayscale: use directly.
    # If RGB/RGBA: use luminance (this is only "correct" for grayscale images, but is best-effort).
    if Af.ndim == 2:
        M = Af
    else:
        rgb = Af[..., :3]
        M = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        if Af.shape[-1] == 4:
            alpha = Af[..., 3]
            M = M * alpha  # composite onto black
    M = np.clip(M, 0.0, 1.0)

    # optional polarity
    X = (1.0 - M) if config['reconstruction']['invert'] else M
    X = np.flipud(X)  # or: X = X[::-1, :]

    # --- map intensity -> dB and then to amplitude mel ---
    S_db = X * config['reconstruction']['top_db'] - config['reconstruction']['top_db']
    S_db = np.maximum(S_db, config['reconstruction']['floor_db'])
    S_mel = librosa.db_to_amplitude(S_db)

    y = librosa.feature.inverse.mel_to_audio(
        S_mel,
        sr=config['audio']['sample_rate'],
        n_fft=config['mel_spectrogram']['n_fft'],
        hop_length=config['mel_spectrogram']['hop_length'],
        n_iter=config['reconstruction']['n_iter'],
        fmin=config['mel_spectrogram']['fmin'],
        fmax=config['mel_spectrogram']['fmax'],
        power=config['mel_spectrogram']['power'],
    )

    # force exact length if requested
    if config['reconstruction']['target_seconds'] is not None:
        target_len = int(round(config['reconstruction']['target_seconds'] * config['audio']['sample_rate']))
        if len(y) >= target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))

    # normalize volume
    y = y / np.max(np.abs(y)+1e-6)

    sf.write(audio_output_path, y, samplerate = config['audio']['sample_rate'])