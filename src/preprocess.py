from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

GENRE_GROUPS = {
    'acoustic': 'folk_acoustic_country',
    'afrobeat': 'world_regional',
    'alt-rock': 'rock_punk_alternative',
    'alternative': 'rock_punk_alternative',
    'ambient': 'ambient_chill_study',
    'anime': 'pop_global',
    'black-metal': 'metal_hardcore',
    'bluegrass': 'folk_acoustic_country',
    'blues': 'jazz_blues_soul_funk',
    'brazil': 'latin_brazilian',
    'breakbeat': 'electronic_dance',
    'british': 'pop_global',
    'cantopop': 'pop_global',
    'chicago-house': 'electronic_dance',
    'children': 'children_show_film',
    'chill': 'ambient_chill_study',
    'classical': 'classical_instrumental',
    'club': 'electronic_dance',
    'comedy': 'children_show_film',
    'country': 'folk_acoustic_country',
    'dance': 'electronic_dance',
    'dancehall': 'reggae_dancehall',
    'death-metal': 'metal_hardcore',
    'deep-house': 'electronic_dance',
    'detroit-techno': 'electronic_dance',
    'disco': 'electronic_dance',
    'disney': 'children_show_film',
    'drum-and-bass': 'electronic_dance',
    'dub': 'reggae_dancehall',
    'dubstep': 'electronic_dance',
    'edm': 'electronic_dance',
    'electro': 'electronic_dance',
    'electronic': 'electronic_dance',
    'emo': 'rock_punk_alternative',
    'folk': 'folk_acoustic_country',
    'forro': 'latin_brazilian',
    'french': 'world_regional',
    'funk': 'jazz_blues_soul_funk',
    'garage': 'electronic_dance',
    'german': 'world_regional',
    'gospel': 'jazz_blues_soul_funk',
    'goth': 'rock_punk_alternative',
    'grindcore': 'metal_hardcore',
    'groove': 'jazz_blues_soul_funk',
    'grunge': 'rock_punk_alternative',
    'guitar': 'classical_instrumental',
    'happy': 'mood_lifestyle',
    'hard-rock': 'rock_punk_alternative',
    'hardcore': 'metal_hardcore',
    'hardstyle': 'electronic_dance',
    'heavy-metal': 'metal_hardcore',
    'hip-hop': 'hiphop_rnb',
    'honky-tonk': 'folk_acoustic_country',
    'house': 'electronic_dance',
    'idm': 'electronic_dance',
    'indian': 'world_regional',
    'indie': 'rock_punk_alternative',
    'indie-pop': 'pop_global',
    'industrial': 'electronic_dance',
    'iranian': 'world_regional',
    'j-dance': 'pop_global',
    'j-idol': 'pop_global',
    'j-pop': 'pop_global',
    'j-rock': 'rock_punk_alternative',
    'jazz': 'jazz_blues_soul_funk',
    'k-pop': 'pop_global',
    'kids': 'children_show_film',
    'latin': 'latin_brazilian',
    'latino': 'latin_brazilian',
    'malay': 'world_regional',
    'mandopop': 'pop_global',
    'metal': 'metal_hardcore',
    'metalcore': 'metal_hardcore',
    'minimal-techno': 'electronic_dance',
    'mpb': 'latin_brazilian',
    'new-age': 'ambient_chill_study',
    'opera': 'classical_instrumental',
    'pagode': 'latin_brazilian',
    'party': 'mood_lifestyle',
    'piano': 'classical_instrumental',
    'pop': 'pop_global',
    'pop-film': 'children_show_film',
    'power-pop': 'pop_global',
    'progressive-house': 'electronic_dance',
    'psych-rock': 'rock_punk_alternative',
    'punk': 'rock_punk_alternative',
    'punk-rock': 'rock_punk_alternative',
    'r-n-b': 'hiphop_rnb',
    'reggae': 'reggae_dancehall',
    'reggaeton': 'latin_brazilian',
    'rock': 'rock_punk_alternative',
    'rock-n-roll': 'rock_punk_alternative',
    'rockabilly': 'folk_acoustic_country',
    'romance': 'mood_lifestyle',
    'sad': 'mood_lifestyle',
    'salsa': 'latin_brazilian',
    'samba': 'latin_brazilian',
    'sertanejo': 'latin_brazilian',
    'show-tunes': 'children_show_film',
    'singer-songwriter': 'folk_acoustic_country',
    'ska': 'reggae_dancehall',
    'sleep': 'ambient_chill_study',
    'songwriter': 'folk_acoustic_country',
    'soul': 'jazz_blues_soul_funk',
    'spanish': 'latin_brazilian',
    'study': 'ambient_chill_study',
    'swedish': 'pop_global',
    'synth-pop': 'pop_global',
    'tango': 'latin_brazilian',
    'techno': 'electronic_dance',
    'trance': 'electronic_dance',
    'trip-hop': 'electronic_dance',
    'turkish': 'world_regional',
    'world-music': 'world_regional',
}

def preprocess_data(df, feature_cols, group_genres=True, verbose=True):
    df = df.copy()
    target_col = 'genre_group' if group_genres else 'track_genre'

    if group_genres:
        df[target_col] = df['track_genre'].map(GENRE_GROUPS).fillna('other')

    X = df[feature_cols].astype(float).to_numpy()
    y = df[target_col].to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        train_size=0.8,
        random_state=25,
        shuffle=True,
        stratify=y
    )

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train_raw = y[train_idx]
    y_test_raw = y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_scaled = scaler.transform(X)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    
    if verbose:
        print("Preprocessing complete")
        print("Target column:", target_col)
        print("Number of target classes:", len(encoder.classes_))
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, scaler, encoder, X_scaled, df

def compute_training_class_weights(y_train):
    class_labels = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=y_train
    )

    class_weight_dict = dict(zip(class_labels, class_weights))

    print("Class weights computed successfully")
    print(f"Number of classes: {len(class_weights)}")
    print(f"Min weight: {class_weights.min():.4f}")
    print(f"Max weight: {class_weights.max():.4f}")

    return class_weight_dict
