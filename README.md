# Music Prediction

This project uses Spotify track metadata in `data/dataset.csv` to load, preprocess, train, and recommend tracks.

## Project Structure

```text
project/
├── data/
│   └── dataset.csv
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── recommend.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/main.py
```

This trains the model and saves song embeddings to `data/song_embeddings.npz`.

## Recommend Without Retraining

```bash
python src/run_recommendation.py "Blinding Lights"
```

You can change how many songs are shown:

```bash
python src/run_recommendation.py "Blinding Lights" --top-n 10
```

The recommendation score now favors neural embedding similarity, with smaller
weights for raw audio feature similarity and popularity. It also gives a boost
to songs from the same broad genre group and the same original Spotify genre,
which helps niche songs surface when they are musically close. You can tune that
blend from the command line:

```bash
python src/run_recommendation.py "Blinding Lights" --popularity-weight 0.05 --feature-weight 0.25 --embedding-weight 0.55 --genre-weight 0.10 --exact-genre-weight 0.05
```

## Simple Frontend

```bash
python web_app.py
```

Then open `http://127.0.0.1:8787`. The page uses the saved embeddings file,
so run `python src/main.py` once first if `data/song_embeddings.npz` is missing.
