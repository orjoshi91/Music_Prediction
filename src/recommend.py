import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_tracks(song_name, embeddings, df, top_n=5):

    matches = df[
        df["track_name"]
        .str.lower()
        .str.contains(song_name.lower(), na=False)
    ]

    if len(matches) == 0:
        print("Song not found.")
        return None

    song_index = matches.index[0]

    similarities = cosine_similarity(
        embeddings[song_index].reshape(1, -1),
        embeddings
    )[0]

    similarities[song_index] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]

    recommendations = df.iloc[top_indices][
        ["track_name", "artists", "track_genre"]
    ]

    return recommendations
