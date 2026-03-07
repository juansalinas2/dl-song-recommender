import numpy as np

from song_recommender.data import load_png_resized, load_raw_resized

def baseline_embedding(spec_path_list, config) -> np.ndarray:
    if config['baseline_embedding']['image_flag'] == True:
        chans = [load_png_resized(path, image_size=config['baseline_embedding']['image_size']) 
                 for path in spec_path_list]
    else:
        chans = [load_raw_resized(path, config['baseline_embedding']['image_size']) 
                 for path in spec_path_list]

    x = np.stack(chans, axis=0)  # (C, H, W)
    
    return x.reshape(-1)