import argparse
from pathlib import Path

import numpy as np

from evaluate_embeddings import load_embedding_artifact
from load_data import load_dataset
from preprocess import preprocess_data
from recommend import print_recommendations, recommend_tracks


def align_saved_embeddings(embeddings, track_ids, processed_df):
    saved_track_ids = np.asarray(track_ids).astype(str)
    current_track_ids = processed_df["track_id"].astype(str).to_numpy()

    if len(saved_track_ids) != len(current_track_ids):
        raise ValueError(
            "Saved embeddings do not match the current processed dataset size. "
            "Run src/main.py again to regenerate embeddings."
        )

    if np.array_equal(saved_track_ids, current_track_ids):
        return embeddings, processed_df

    indexed_df = processed_df.set_index(
        processed_df["track_id"].astype(str),
        drop=False
    )

    missing_ids = set(saved_track_ids) - set(indexed_df.index)
    if missing_ids:
        raise ValueError(
            "Saved embeddings contain songs that are not in the current dataset. "
            "Run src/main.py again to regenerate embeddings."
        )

    aligned_df = indexed_df.loc[saved_track_ids].reset_index(drop=True)
    return embeddings, aligned_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend songs from saved embeddings without retraining."
    )
    parser.add_argument(
        "song_name",
        nargs="?",
        default="Blinding Lights",
        help="Song title to search for."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of recommendations to show."
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.45,
        help="Weight for neural embedding similarity."
    )
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=0.30,
        help="Weight for raw audio feature similarity."
    )
    parser.add_argument(
        "--popularity-weight",
        type=float,
        default=0.20,
        help="Weight for similar popularity."
    )
    parser.add_argument(
        "--genre-weight",
        type=float,
        default=0.05,
        help="Weight for matching genre group."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset.csv"
    embeddings_path = project_root / "data" / "song_embeddings.npz"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run src/main.py once to train and save "
            "embeddings before using recommendations."
        )

    df, feature_cols = load_dataset(dataset_path, verbose=False)
    *_, X_all, processed_df = preprocess_data(df, feature_cols, verbose=False)
    embeddings, track_ids, _ = load_embedding_artifact(embeddings_path)
    embeddings, processed_df = align_saved_embeddings(
        embeddings,
        track_ids,
        processed_df
    )

    source_song, recommendations = recommend_tracks(
        args.song_name,
        embeddings,
        processed_df,
        top_n=args.top_n,
        feature_matrix=X_all,
        score_weights={
            "embedding": args.embedding_weight,
            "features": args.feature_weight,
            "popularity": args.popularity_weight,
            "genre": args.genre_weight,
        }
    )
    print_recommendations(source_song, recommendations)


if __name__ == "__main__":
    main()
