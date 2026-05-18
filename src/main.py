from load_data import load_dataset
from preprocess import preprocess_data
from train import train_model
from recommend import recommend_tracks


def main():
    df, feature_cols = load_dataset("data/dataset.csv")
    X_train, X_test, y_train, y_test, scalar, encoder = preprocess_data(df, feature_cols)
    # model = train_model(processed_data)
    num_classes = len(encoder.classes_)

    model, history = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        num_classes
    )
    # recommendations = recommend_tracks(model, processed_data)
    #
    # print(recommendations)
    pass


main()
