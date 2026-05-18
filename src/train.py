import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model(X_train, X_test, y_train, y_test, num_classes):

    model = Sequential()

    # Input layer
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

    model.add(Dense(32, activation='relu'))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model...\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Test Accuracy:", accuracy)

    return model, history
