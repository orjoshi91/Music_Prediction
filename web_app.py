import json
import errno
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from evaluate_embeddings import load_embedding_artifact
from load_data import load_dataset
from preprocess import preprocess_data
from recommend import recommend_tracks

HOST = "127.0.0.1"
PORT = 8787
TOP_SONG_LIMIT = 500


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Music Recommender</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6871;
      --line: #d9e0e6;
      --panel: #ffffff;
      --bg: #f5f7f9;
      --accent: #0f766e;
      --accent-dark: #0b5d56;
      --soft: #e8f3f1;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }

    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }

    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 22px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .status {
      color: var(--muted);
      font-size: 14px;
      text-align: right;
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(180px, 0.8fr) minmax(280px, 1.4fr) auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 24px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
      color: #2f3b43;
    }

    select,
    button {
      width: 100%;
      min-height: 42px;
      border-radius: 6px;
      font-size: 15px;
    }

    select {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 0 12px;
    }

    button {
      border: 0;
      background: var(--accent);
      color: white;
      font-weight: 700;
      padding: 0 18px;
      cursor: pointer;
    }

    button:hover {
      background: var(--accent-dark);
    }

    button:disabled,
    select:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .source {
      padding: 14px 0;
      margin-bottom: 6px;
      color: var(--muted);
    }

    .source strong {
      color: var(--ink);
    }

    .results {
      display: grid;
      gap: 10px;
    }

    .result {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }

    .result-top {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: start;
    }

    .title {
      font-weight: 800;
      line-height: 1.25;
    }

    .artist {
      margin-top: 3px;
      color: var(--muted);
      line-height: 1.35;
    }

    .score {
      font-weight: 800;
      color: var(--accent-dark);
      white-space: nowrap;
    }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }

    .metric {
      background: var(--soft);
      color: #16413d;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 700;
    }

    .message {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
      color: var(--muted);
    }

    @media (max-width: 760px) {
      header {
        display: block;
      }

      .status {
        margin-top: 8px;
        text-align: left;
      }

      .controls {
        grid-template-columns: 1fr;
      }

      .result-top {
        display: block;
      }

      .score {
        margin-top: 8px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Music Recommender</h1>
      <div class="status" id="status">Loading saved embeddings...</div>
    </header>

    <section class="controls">
      <div>
        <label for="genreSelect">Genre group</label>
        <select id="genreSelect" disabled></select>
      </div>
      <div>
        <label for="songSelect">Song</label>
        <select id="songSelect" disabled></select>
      </div>
      <button id="recommendButton" disabled>Recommend</button>
    </section>

    <section id="output" class="message">Select a genre and song.</section>
  </main>

  <script>
    const genreSelect = document.getElementById("genreSelect");
    const songSelect = document.getElementById("songSelect");
    const recommendButton = document.getElementById("recommendButton");
    const output = document.getElementById("output");
    const statusNode = document.getElementById("status");

    function setMessage(text) {
      output.className = "message";
      output.textContent = text;
    }

    async function getJson(url) {
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    function option(label, value) {
      const node = document.createElement("option");
      node.textContent = label;
      node.value = value;
      return node;
    }

    async function loadGenres() {
      const data = await getJson("/api/genres");
      statusNode.textContent = data.song_count.toLocaleString() + " songs loaded";
      genreSelect.replaceChildren(
        option("Select genre group", ""),
        ...data.genres.map((genre) => option(genre.label, genre.value))
      );
      genreSelect.disabled = false;
    }

    async function loadSongs() {
      const genre = genreSelect.value;
      songSelect.replaceChildren(option("Select song", ""));
      songSelect.disabled = true;
      recommendButton.disabled = true;
      setMessage("Select a song.");

      if (!genre) {
        setMessage("Select a genre and song.");
        return;
      }

      const data = await getJson("/api/songs?genre=" + encodeURIComponent(genre));
      songSelect.replaceChildren(
        option("Select song", ""),
        ...data.songs.map((song) => {
          const label = song.track_name + " - " + song.artists +
            " (popularity " + song.popularity + ")";
          return option(label, song.track_id);
        })
      );
      songSelect.disabled = false;
    }

    function renderRecommendations(data) {
      output.className = "";
      output.replaceChildren();

      const source = document.createElement("div");
      source.className = "source";
      source.innerHTML = "<strong>" + data.source.track_name + "</strong> by " +
        data.source.artists + "<br>Original genre: " + data.source.track_genre +
        " | Genre group: " + data.source.genre_group;
      output.appendChild(source);

      const results = document.createElement("div");
      results.className = "results";

      data.recommendations.forEach((song, index) => {
        const item = document.createElement("article");
        item.className = "result";
        item.innerHTML =
          "<div class=\\"result-top\\">" +
            "<div>" +
              "<div class=\\"title\\">" + (index + 1) + ". " + song.track_name + "</div>" +
              "<div class=\\"artist\\">" + song.artists + "</div>" +
            "</div>" +
            "<div class=\\"score\\">Score " + song.score.toFixed(3) + "</div>" +
          "</div>" +
          "<div class=\\"metrics\\">" +
            "<span class=\\"metric\\">Embedding " + song.embedding.toFixed(3) + "</span>" +
            "<span class=\\"metric\\">Features " + song.features.toFixed(3) + "</span>" +
            "<span class=\\"metric\\">Popularity " + song.popularity_match.toFixed(3) + "</span>" +
            "<span class=\\"metric\\">" + song.track_genre + "</span>" +
            "<span class=\\"metric\\">" + song.genre_group + "</span>" +
          "</div>";
        results.appendChild(item);
      });

      output.appendChild(results);
    }

    async function recommend() {
      const trackId = songSelect.value;
      if (!trackId) {
        return;
      }

      recommendButton.disabled = true;
      setMessage("Finding recommendations...");

      try {
        const data = await getJson("/api/recommend?track_id=" + encodeURIComponent(trackId));
        renderRecommendations(data);
      } catch (error) {
        setMessage(error.message);
      } finally {
        recommendButton.disabled = false;
      }
    }

    genreSelect.addEventListener("change", loadSongs);
    songSelect.addEventListener("change", () => {
      recommendButton.disabled = !songSelect.value;
    });
    recommendButton.addEventListener("click", recommend);

    loadGenres().catch((error) => {
      statusNode.textContent = "Embeddings unavailable";
      setMessage(error.message);
    });
  </script>
</body>
</html>
"""


class RecommenderData:
    def __init__(self):
        self.loaded = False
        self.error = None

    def load(self):
        if self.loaded:
            return

        dataset_path = PROJECT_ROOT / "data" / "dataset.csv"
        embeddings_path = PROJECT_ROOT / "data" / "song_embeddings.npz"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                "Missing data/song_embeddings.npz. Run python3 src/main.py once "
                "to train and save embeddings."
            )

        df, feature_cols = load_dataset(dataset_path, verbose=False)
        *_, feature_matrix, processed_df = preprocess_data(
            df,
            feature_cols,
            verbose=False
        )
        embeddings, track_ids, _ = load_embedding_artifact(embeddings_path)
        self.embeddings, self.feature_matrix, self.df = self.align_saved_embeddings(
            embeddings,
            track_ids,
            feature_matrix,
            processed_df
        )
        self.loaded = True

    def align_saved_embeddings(
        self,
        embeddings,
        track_ids,
        feature_matrix,
        processed_df
    ):
        saved_track_ids = np.asarray(track_ids).astype(str)
        current_track_ids = processed_df["track_id"].astype(str).to_numpy()

        if len(saved_track_ids) != len(current_track_ids):
            raise ValueError(
                "Saved embeddings do not match the current dataset size. "
                "Run python3 src/main.py again to regenerate embeddings."
            )

        if np.array_equal(saved_track_ids, current_track_ids):
            return embeddings, feature_matrix, processed_df.reset_index(drop=True)

        positions = {
            track_id: position
            for position, track_id in enumerate(current_track_ids)
        }
        missing_ids = [track_id for track_id in saved_track_ids if track_id not in positions]

        if missing_ids:
            raise ValueError(
                "Saved embeddings do not match the current dataset. "
                "Run python3 src/main.py again to regenerate embeddings."
            )

        order = [positions[track_id] for track_id in saved_track_ids]
        aligned_df = processed_df.iloc[order].reset_index(drop=True)
        aligned_features = feature_matrix[order]
        return embeddings, aligned_features, aligned_df

    def genres(self):
        self.load()
        counts = self.df["genre_group"].value_counts().sort_index()
        return [
            {
                "value": genre,
                "label": f"{genre} ({count})",
                "count": int(count),
            }
            for genre, count in counts.items()
        ]

    def songs_for_genre(self, genre):
        self.load()
        songs = self.df[self.df["genre_group"] == genre].copy()

        if songs.empty:
            return []

        songs = songs.sort_values(
            ["popularity", "track_name"],
            ascending=[False, True]
        ).head(TOP_SONG_LIMIT)

        return [
            {
                "track_id": str(song["track_id"]),
                "track_name": song["track_name"],
                "artists": song["artists"],
                "track_genre": song["track_genre"],
                "popularity": int(song["popularity"]),
            }
            for _, song in songs.iterrows()
        ]

    def recommendations(self, track_id):
        self.load()
        source_song, recommendations = recommend_tracks(
            "",
            self.embeddings,
            self.df,
            top_n=5,
            feature_matrix=self.feature_matrix,
            source_track_id=track_id
        )

        if source_song is None or recommendations is None:
            raise ValueError("Selected song was not found.")

        return {
            "source": serialize_song(source_song),
            "recommendations": [
                serialize_song(song)
                for _, song in recommendations.iterrows()
            ],
        }


def serialize_song(song):
    output = {
        "track_id": str(song.get("track_id", "")),
        "track_name": song["track_name"],
        "artists": song["artists"],
        "track_genre": song["track_genre"],
        "genre_group": song["genre_group"],
    }

    for key in ["score", "embedding", "features", "popularity_match"]:
        if key in song:
            output[key] = float(song[key])

    if "popularity" in song:
        output["popularity"] = int(song["popularity"])

    return output


DATA = RecommenderData()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/":
                self.send_html(HTML)
            elif parsed.path == "/api/genres":
                DATA.load()
                self.send_json({
                    "genres": DATA.genres(),
                    "song_count": int(len(DATA.df)),
                })
            elif parsed.path == "/api/songs":
                params = parse_qs(parsed.query)
                genre = params.get("genre", [""])[0]
                self.send_json({"songs": DATA.songs_for_genre(genre)})
            elif parsed.path == "/api/recommend":
                params = parse_qs(parsed.query)
                track_id = params.get("track_id", [""])[0]
                if not track_id:
                    raise ValueError("Missing track_id.")
                self.send_json(DATA.recommendations(track_id))
            else:
                self.send_error(404)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, status=500)

    def log_message(self, format, *args):
        return

    def send_html(self, html):
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    try:
        server = ReusableThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Music recommender frontend is already running at http://{HOST}:{PORT}")
            return
        raise

    print(f"Music recommender frontend running at http://{HOST}:{PORT}")
    print("Use python3 src/main.py first if data/song_embeddings.npz is missing.")
    server.serve_forever()


if __name__ == "__main__":
    main()
