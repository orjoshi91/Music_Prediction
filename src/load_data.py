import pandas as pd

def load_dataset(filepath, verbose=True):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(filepath)

    # These are the numeric track features we will use to train our model
    feature_cols = [
        'popularity',
        'duration_ms',
        'explicit',
        'danceability',     # How suitable a track is for dancing
        'energy',           # Intensity and activity of the track
        'key',
        'loudness',         # Overall loudness in decibels
        'mode',
        'speechiness',      # Presence of spoken words
        'acousticness',     # How acoustic the track is
        'instrumentalness', # Whether a track has no vocals
        'liveness',         # Presence of a live audience
        'valence',          # Musical positiveness of the track
        'tempo',            # Speed of the track in BPM
        'time_signature'
    ]

    # Remove rows where any feature value or genre is missing
    df = df.dropna(subset=feature_cols + ['track_genre'])

    # Remove duplicate songs keeping only the first occurrence
    df = df.drop_duplicates(subset='track_id')

    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Number of genres: {df['track_genre'].nunique()}")
        print(f"Sample:\n{df[feature_cols].head()}")

    return df, feature_cols
