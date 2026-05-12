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

## Run

```bash
python src/main.py
```
