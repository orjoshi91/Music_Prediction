import numpy as np
import tensorflow as tf


def get_embedding_model(model, embedding_layer_name='song_embedding'):
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(embedding_layer_name).output
    )


def generate_embeddings(model, X, embedding_layer_name='song_embedding'):
    embedding_model = get_embedding_model(model, embedding_layer_name)
    return embedding_model.predict(X, verbose=0)


def save_embeddings(filepath, embeddings, track_ids=None, labels=None):
    payload = {'embeddings': embeddings}

    if track_ids is not None:
        payload['track_ids'] = np.asarray(track_ids)

    if labels is not None:
        payload['labels'] = np.asarray(labels)

    np.savez_compressed(filepath, **payload)
