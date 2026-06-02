from pathlib import Path

from embeddings import generate_embeddings, save_embeddings
from evaluate_embeddings import evaluate_embedding_neighbors
from load_data import load_dataset
from preprocess import compute_training_class_weights, preprocess_data
from train import train_model
from recommend import recommend_tracks


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "dataset.csv"
    embeddings_path = project_root / "data" / "song_embeddings.npz"

    df, feature_cols = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test, scaler, encoder, X_all, processed_df = preprocess_data(
        df,
        feature_cols
    )
    class_weights = compute_training_class_weights(y_train)
    # model = train_model(processed_data)
    num_classes = len(encoder.classes_)

    model, history, metrics = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        num_classes,
        class_weights,
        architecture='tuned'
    )

    embeddings = generate_embeddings(model, X_all)
    save_embeddings(
        embeddings_path,
        embeddings,
        track_ids=processed_df['track_id'],
        labels=processed_df['genre_group']
    )
    recommendation_metrics = evaluate_embedding_neighbors(
        embeddings,
        processed_df['genre_group']
    )

    print(f"Saved embeddings to {embeddings_path}")
    print(f"Best validation accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"Precision@5: {recommendation_metrics['precision_at_5']:.4f}")
    print(f"Hit rate@5: {recommendation_metrics['hit_rate_at_5']:.4f}")
    print(f"Precision@10: {recommendation_metrics['precision_at_10']:.4f}")
    print(f"Hit rate@10: {recommendation_metrics['hit_rate_at_10']:.4f}")
    
    print("\nTop Song Recommendations:\n")

    recommendations = recommend_tracks(
        "Blinding Lights",
        embeddings,
        processed_df,
        top_n=5
    )

    print(recommendations)


if __name__ == "__main__":
    main()
