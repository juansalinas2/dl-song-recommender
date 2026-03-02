# save mel spectrograms for all audio in a list of audio_paths
# reference for max power is set based on first track in list

import librosa
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os

from song_recommender.paths import *

def generate_mel_spectrograms(audio_paths, config):
    S_list = []

    for i in range(len(audio_paths)):
        # load audio
        y, sr = librosa.load(
            audio_paths[i],
            sr=config['audio']['sample_rate'],
            duration=config['audio']['duration']
        )

        # generate mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=config['mel_spectrogram']['n_fft'],
            hop_length=config['mel_spectrogram']['hop_length'],
            n_mels=config['mel_spectrogram']['n_mels'],
            fmin=config['mel_spectrogram']['fmin'],
            fmax=config['mel_spectrogram']['fmax'],
            power=config['mel_spectrogram']['power'],
        )

        # important for spectrograms for stems!
        if i == 0:
            # set maximum power reference for track and stems
            # note: we need to set 0dB using original audio
            # to avoid power_to_db scaling silence 
            # and low-level noise in stems
            # drawback: losing loudness distinction between songs
            # i.e., loses "absolute" energy which is already sensitive to recording quality
            ref_songmax = np.max(S) 

        # convert to dB
        S_db = librosa.power_to_db(S, ref=ref_songmax)

        # min-max normalize: want decibel range in S_dB to be [-80,0] -> [0,1]
        if config['normalization']['method'] == 'minmax':
            S_norm = (S_db + 80) / 80
            # clip invalid values outside [0,1] -- negative values that are unnecessary and likely noise
            if config['normalization']['clip'] == True:
                S_list += [np.clip(S_norm, 0, 1)]
            else:
                S_list += [S_norm]
        else:
            S_list += [S_db]

    return S_list

def save_mel_spectrogram(S_norm, 
                         track_id : str, 
                         output_name : str,
                         config,
                         png_output_base : Path = SPECTROGRAM_PNG_DIR,
                         raw_output_base : Path = SPECTROGRAM_RAW_DIR):
    
    png_output_base = png_output_base / track_id
    raw_output_base = raw_output_base / track_id
    png_output_path = png_output_base / output_name
    raw_output_path = raw_output_base / output_name

    # if necessary, create directory to save png 
    if not os.path.exists(png_output_base):
            os.makedirs(png_output_base)
            print(f'Created subdirectory {png_output_base} in current directory to save generated pngs')

    # save spectrogram .png - 8-bit, colormapped, 
    plt.imsave(
       f'{png_output_path}.{config['image_output']['format']}',
        np.flipud(S_norm), # correct orientation - (0,0) is top left for plt
        cmap=config['image_output']['cmap'],
        vmin=config['image_output']['vmin'], 
        vmax=config['image_output']['vmax']  
        )

    # if necessary, create directory to save npy 
    if not os.path.exists(raw_output_base):
            os.makedirs(raw_output_base)
            print(f'Created subdirectory {raw_output_base} in current directory to save generated npys')

    # save raw spectrogram data
    np.save(f'{raw_output_path}.{config['raw_output']['format']}', 
            S_norm.astype(np.float32))
