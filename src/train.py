import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_size, num_classes, architecture='tuned'):
    tf.keras.utils.set_random_seed(25)

    model = Sequential(name=f'{architecture}_genre_classifier')
    model.add(Input(shape=(input_size,)))

    if architecture == 'baseline':
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu', name='song_embedding'))
        learning_rate = 0.001

    else:
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.20))

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.10))

        model.add(Dense(64, activation='relu'))
        model.add(Dense(48, activation='relu', name='song_embedding'))

        learning_rate = 0.0007

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    num_classes,
    class_weights=None,
    architecture='tuned',
    epochs=40,
    batch_size=128
):

    model = build_model(X_train.shape[1], num_classes, architecture)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]

    print(f"Training {architecture} model...\n")
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    loss, accuracy = model.evaluate(X_test, y_test)

    best_val_accuracy = max(history.history.get('val_accuracy', [accuracy]))
    best_epoch = history.history.get('val_accuracy', [accuracy]).index(best_val_accuracy) + 1

    metrics = {
        'test_loss': loss,
        'test_accuracy': accuracy,
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': best_epoch
    }

    print("\nBest validation accuracy:", best_val_accuracy)
    print("Best epoch:", best_epoch)
    print("Test accuracy:", accuracy)

    return model, history, metrics
