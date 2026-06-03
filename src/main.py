from pathlib import Path

import numpy as np
from embeddings import generate_embeddings, save_embeddings
from evaluate_embeddings import evaluate_embedding_neighbors
from load_data import load_dataset
from preprocess import compute_training_class_weights, preprocess_data
from train import train_model
from recommend import print_recommendations, recommend_tracks
from sklearn.metrics import classification_report, confusion_matrix


def print_per_genre_metrics(model, X_test, y_test, encoder):
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    print("\nPer-Genre Classification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            digits=3,
            zero_division=0
        )
    )

    matrix = confusion_matrix(y_test, y_pred)
    mistakes = []

    for actual_index, actual_label in enumerate(encoder.classes_):
        for predicted_index, predicted_label in enumerate(encoder.classes_):
            if actual_index != predicted_index:
                mistakes.append((
                    matrix[actual_index][predicted_index],
                    actual_label,
                    predicted_label
                ))

    print("Most Common Genre Confusions:")
    for count, actual_label, predicted_label in sorted(mistakes, reverse=True)[:10]:
        if count > 0:
            print(f"- {actual_label} predicted as {predicted_label}: {count}")


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

    print_per_genre_metrics(model, X_test, y_test, encoder)

    source_song, recommendations = recommend_tracks(
        "Blinding Lights",
        embeddings,
        processed_df,
        top_n=5,
        feature_matrix=X_all
    )

    print_recommendations(source_song, recommendations)


if __name__ == "__main__":
    main()
