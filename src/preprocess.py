from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df,feature_cols):
    X = df[feature_cols]
    y = df['track_genre']

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scale,y_int,test_size=0.2,
        train_size = 0.8,
        random_state = 25,
        shuffle = True,
        stratify = y_int
        )
    
    print("Preprocessing complete")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, scaler, encoder
