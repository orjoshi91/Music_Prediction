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

The recommendation score blends neural embedding similarity, raw audio feature
similarity, popularity similarity, and a small genre-group bonus. You can tune
that blend from the command line:

```bash
python src/run_recommendation.py "Blinding Lights" --popularity-weight 0.30 --feature-weight 0.30 --embedding-weight 0.35 --genre-weight 0.05
```
