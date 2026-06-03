import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SCORE_WEIGHTS = {
    "embedding": 0.45,
    "features": 0.30,
    "popularity": 0.20,
    "genre": 0.05,
}


def _normalize_song_name(song_name):
    return song_name.casefold().strip()


def _min_max_scale(values):
    values = np.asarray(values, dtype=float)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    if max_value == min_value:
        return np.ones_like(values)

    return (values - min_value) / (max_value - min_value)


def recommend_tracks(
    song_name,
    embeddings,
    df,
    top_n=5,
    feature_matrix=None,
    score_weights=None
):
    if len(embeddings) != len(df):
        raise ValueError(
            "Embeddings and dataframe must have the same number of songs."
        )

    if feature_matrix is not None and len(feature_matrix) != len(df):
        raise ValueError(
            "Feature matrix and dataframe must have the same number of songs."
        )

    weights = DEFAULT_SCORE_WEIGHTS.copy()
    if score_weights is not None:
        weights.update(score_weights)

    search_name = _normalize_song_name(song_name)
    normalized_track_names = df["track_name"].fillna("").map(_normalize_song_name)

    matches = df[normalized_track_names == search_name]

    if len(matches) == 0:
        matches = df[normalized_track_names.str.contains(search_name, na=False)]

    if len(matches) == 0:
        print("Song not found.")
        return None, None

    if "popularity" in matches.columns:
        source_song = matches.sort_values("popularity", ascending=False).iloc[0]
    else:
        source_song = matches.iloc[0]

    song_position = df.index.get_loc(source_song.name)

    embedding_similarity = cosine_similarity(
        embeddings[song_position].reshape(1, -1),
        embeddings
    )[0]

    final_score = weights["embedding"] * _min_max_scale(embedding_similarity)

    if feature_matrix is not None:
        feature_similarity = cosine_similarity(
            feature_matrix[song_position].reshape(1, -1),
            feature_matrix
        )[0]
        final_score += weights["features"] * _min_max_scale(feature_similarity)
    else:
        feature_similarity = np.zeros(len(df))

    if "popularity" in df.columns:
        popularity = df["popularity"].astype(float).to_numpy()
        source_popularity = float(source_song["popularity"])
        popularity_similarity = 1 - np.abs(popularity - source_popularity) / 100
        popularity_similarity = np.clip(popularity_similarity, 0, 1)
        final_score += weights["popularity"] * popularity_similarity
    else:
        popularity_similarity = np.zeros(len(df))

    if "genre_group" in df.columns:
        genre_bonus = (
            df["genre_group"].to_numpy() == source_song["genre_group"]
        ).astype(float)
        final_score += weights["genre"] * genre_bonus
    else:
        genre_bonus = np.zeros(len(df))

    final_score[song_position] = -1

    top_indices = np.argsort(final_score)[::-1][:top_n]

    columns = ["track_name", "artists", "track_genre"]
    if "genre_group" in df.columns:
        columns.append("genre_group")

    recommendations = df.iloc[top_indices][columns].copy()
    recommendations.insert(0, "score", final_score[top_indices])
    recommendations.insert(1, "embedding", embedding_similarity[top_indices])

    if feature_matrix is not None:
        recommendations.insert(2, "features", feature_similarity[top_indices])

    if "popularity" in df.columns:
        recommendations.insert(
            3 if feature_matrix is not None else 2,
            "popularity_match",
            popularity_similarity[top_indices]
        )

    return source_song, recommendations


def print_recommendations(source_song, recommendations):
    if source_song is None or recommendations is None:
        return

    print("\nRecommendation Source:")
    print(f"{source_song['track_name']} by {source_song['artists']}")
    print(
        f"Original genre: {source_song['track_genre']} | "
        f"Genre group: {source_song['genre_group']}"
    )

    print("\nTop Song Recommendations:")
    for rank, (_, song) in enumerate(recommendations.iterrows(), start=1):
        score_parts = [
            f"Score: {song['score']:.3f}",
            f"Embedding: {song['embedding']:.3f}",
        ]

        if "features" in song:
            score_parts.append(f"Features: {song['features']:.3f}")

        if "popularity_match" in song:
            score_parts.append(f"Popularity match: {song['popularity_match']:.3f}")

        print(
            f"{rank}. {song['track_name']} by {song['artists']}\n"
            f"   {' | '.join(score_parts)}\n"
            f"   Original genre: {song['track_genre']} | "
            f"Genre group: {song['genre_group']}"
        )
