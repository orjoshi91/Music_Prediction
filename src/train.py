import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def train_model(X_train, X_test, y_train, y_test, num_classes):

    # neural network
    model = Sequential([
        Input(shape=(X_train.shape[1],)),

        Dense(128, activation='relu'),
        Dropout(0.2),

        Dense(64, activation='relu', name='embedding_layer'),

        Dense(num_classes, activation='softmax')
    ])

    # Compiling the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Stoping the training early if validation loss stops improving
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("\nTraining model...\n")

    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluating performance
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("\nTest Accuracy:", accuracy)

    return model, history
