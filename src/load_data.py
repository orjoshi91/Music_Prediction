import pandas as pd

def load_dataset(filepath):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(filepath)

    # These are the audio features we will use to train our model
    feature_cols = [
        'danceability',     # How suitable a track is for dancing
        'energy',           # Intensity and activity of the track
        'loudness',         # Overall loudness in decibels
        'speechiness',      # Presence of spoken words
        'acousticness',     # How acoustic the track is
        'instrumentalness', # Whether a track has no vocals
        'liveness',         # Presence of a live audience
        'valence',          # Musical positiveness of the track
        'tempo'             # Speed of the track in BPM
    ]

    # Remove rows where any feature value or genre is missing
    df = df.dropna(subset=feature_cols + ['track_genre'])

    # Remove duplicate songs keeping only the first occurrence
    df = df.drop_duplicates(subset='track_id')

    # Print a summary so we can verify the data loaded correctly
    print(f"Dataset shape: {df.shape}")
    print(f"Number of genres: {df['track_genre'].nunique()}")
    print(f"Sample:\n{df[feature_cols].head()}")

    return df, feature_cols

# Call the function with the path to our dataset
df, feature_cols = load_dataset('data/dataset.csv')