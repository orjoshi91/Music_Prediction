from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_embedding_artifact(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data['embeddings'], data['track_ids'], data['labels']


def evaluate_embedding_neighbors(embeddings, labels, k_values=(5, 10), sample_size=5000, random_state=25):
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)

    if sample_size is not None and sample_size < len(embeddings):
        query_indices = rng.choice(len(embeddings), size=sample_size, replace=False)
    else:
        query_indices = np.arange(len(embeddings))

    query_embeddings = embeddings[query_indices]
    similarities = cosine_similarity(query_embeddings, embeddings)
    max_k = max(k_values)

    # Exclude each sampled song from recommending itself.
    similarities[np.arange(len(query_indices)), query_indices] = -np.inf
    neighbor_indices = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]

    metrics = {}
    query_labels = labels[query_indices]

    for k in k_values:
        top_k_indices = neighbor_indices[:, :k]
        top_k_labels = labels[top_k_indices]
        matches = top_k_labels == query_labels[:, None]
        metrics[f'precision_at_{k}'] = float(matches.mean())
        metrics[f'hit_rate_at_{k}'] = float(matches.any(axis=1).mean())

    metrics['sample_size'] = int(len(query_indices))
    return metrics


def main():
    project_root = Path(__file__).resolve().parents[1]
    embeddings_path = project_root / 'data' / 'song_embeddings.npz'
    embeddings, track_ids, labels = load_embedding_artifact(embeddings_path)
    metrics = evaluate_embedding_neighbors(embeddings, labels)

    print(f"Embedding file: {embeddings_path}")
    print(f"Songs evaluated: {metrics['sample_size']}")
    for key, value in metrics.items():
        if key != 'sample_size':
            print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
