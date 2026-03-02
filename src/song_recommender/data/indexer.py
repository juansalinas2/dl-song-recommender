from song_recommender.paths import *

class TrackIndexer:

    # class variables
    stem_list = ['bass','drums','other','vocals']

    ## define how we intialize a TrackIndexer object
    def __init__(self, df, 
                 track_id_label : str = 'spotify_id', # can specify a different identifier
                 audio_dir: Path = AUDIO_DIR,
                 stems_dir: Path = STEMS_DIR,
                 spec_png_dir: Path = SPECTROGRAM_PNG_DIR,
                 spec_raw_dir: Path = SPECTROGRAM_RAW_DIR):
        
        self.df = df
        self.track_id_label = track_id_label
        self.audio_dir = audio_dir
        self.stems_dir = stems_dir
        self.spec_png_dir = spec_png_dir
        self.spec_raw_dir = spec_raw_dir

    def get_track_index(self, track_id):
        return self.df.index.get_loc(
            self.df.loc[self.df[self.track_id_label] == track_id].index[0]
        )

    def get_audio_paths(self, track_id):
        return [self.audio_dir / f"{track_id}.wav"] + [
            self.stems_dir / track_id / f"{stem}.wav"
            for stem in self.stem_list
        ]

    def get_spec_png_paths(self, track_id):
        base = self.spec_png_dir / track_id
        return [base / f"{track_id}.png"] + [
            base / f"{stem}.png"
            for stem in self.stem_list
        ]

    def get_spec_raw_paths(self, track_id):
        base = self.spec_raw_dir / track_id
        return [base / f"{track_id}.npy"] + [
            base / f"{stem}.npy"
            for stem in self.stem_list
        ]
    
    def add_paths_to_df(self):
        self.df['audio_paths'] = self.df[self.track_id_label].apply(self.get_audio_paths)
        self.df['spec_png_paths'] = self.df[self.track_id_label].apply(self.get_spec_png_paths)
        self.df['spec_raw_paths'] = self.df[self.track_id_label].apply(self.get_spec_raw_paths)

        return self.df