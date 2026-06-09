Jackson Horman, Johanna Sanchez, Ovi Joshi
CSE 25
Professor Minnes
1 June 2026

# Music Recommendation Project Final Report

## 1. Introduction

### Problem Statement

Music platforms such as Spotify and Apple Music contain millions of songs across many artists, genres, moods, and popularity levels. This makes it difficult for listeners to discover songs that are similar to music they already enjoy. Our project addresses this problem by building a music recommendation system that learns patterns from Spotify track features and uses those learned patterns to recommend similar songs.

Instead of only matching songs by title, artist, or genre label, our system compares songs using numerical audio features such as danceability, energy, loudness, acousticness, instrumentalness, valence, and tempo. We trained a neural network to predict broader genre groups, then used the model's hidden-layer embeddings as learned song representations. Songs with similar embeddings are treated as musically similar.

### Relevance and Importance

Recommendation systems are important because they shape how users discover media. Apps such as Spotify and Apple Music use recommendation systems for autoplay, radio stations, playlists, and personalized discovery. A useful recommendation system can help users find songs they may not have searched for directly, while still keeping the results related to their existing tastes.

This project is relevant because it models a simplified version of that real-world problem. It also shows why machine learning can be useful for recommendation: raw song features are informative, but learned embeddings can capture relationships between songs more effectively than comparing each feature by hand.

### Related Work

Most large music services use a combination of content-based filtering and collaborative filtering. Collaborative filtering uses user behavior, such as listening history, skips, and likes, to recommend songs that similar users enjoyed. Content-based filtering uses information about the songs themselves, such as audio features, metadata, lyrics, genre, or artist information.

Our project focuses on content-based recommendation because the Kaggle dataset includes track-level Spotify features but does not include user listening histories. This made content-based learning the best fit for our data. We used cosine similarity, a common method for comparing vector representations, to measure how close two songs are in the learned embedding space.

## 2. Data Sourcing and Processing

### Dataset Description

We used the Spotify Tracks Dataset from Kaggle. The original dataset contains 114,000 rows and 114 unique track genres. Each row represents a Spotify track and includes metadata such as track name, artist, popularity, duration, explicit flag, and audio features.

After preprocessing, our working dataset contained 89,741 unique songs. The decrease happened because we removed duplicate tracks based on `track_id`. There were no missing rows in the selected feature columns.

The model used 15 numeric input features:

| Feature | Description |
|---|---|
| popularity | Spotify popularity score |
| duration_ms | Track length in milliseconds |
| explicit | Whether the song is marked explicit |
| danceability | Suitability for dancing |
| energy | Intensity and activity level |
| key | Musical key |
| loudness | Loudness in decibels |
| mode | Major/minor modality |
| speechiness | Presence of spoken words |
| acousticness | Acoustic quality |
| instrumentalness | Likelihood of no vocals |
| liveness | Presence of live audience qualities |
| valence | Musical positivity |
| tempo | Beats per minute |
| time_signature | Meter/time signature |

### Data Preprocessing

Before training, we cleaned and transformed the data so the model could learn from it more effectively. First, we loaded the Kaggle CSV file with pandas. We selected the 15 numeric features listed above and removed duplicate songs using `track_id`. We also checked the selected features and genre labels for missing values.

Next, we grouped the original 114 Spotify genres into 14 broader genre groups. For example, genres such as house, dubstep, EDM, techno, and trance were grouped into `electronic_dance`, while genres such as rock, punk, alternative, and grunge were grouped into `rock_punk_alternative`. This made the classification task more realistic because many of the original genre labels overlap musically.

We then split the data into 80% training and 20% testing using stratified sampling. The final split was 71,792 training songs and 17,949 testing songs. Stratification ensured that each genre group was represented in both sets.

Finally, we normalized the numeric features with scikit-learn's `StandardScaler`. This step centered features around zero and scaled them to comparable ranges. Normalization was important because features such as `duration_ms` and `tempo` have much larger raw values than features such as `danceability` or `valence`. Without scaling, the larger-numbered features could dominate the model.

### Importance of Processing

These preprocessing steps improved the quality and fairness of the model. Removing duplicate songs prevented the model from learning repeated examples. Grouping genres reduced noise from overly specific labels. Stratified splitting made evaluation more representative. Scaling made the input features comparable so that the neural network could learn from all features instead of being biased toward the largest numeric ranges.

We also computed class weights during training to address class imbalance. Some genre groups appear much more often than others, so class weights helped prevent the model from ignoring smaller groups.

## 3. Model Description

### Model Architecture

We used a feedforward neural network built with TensorFlow and Keras. The model takes the 15 normalized Spotify features as input and predicts one of 14 genre groups as output. Although genre prediction is the training task, the main purpose of the model is to learn useful hidden representations of songs. After training, we extract the hidden layer named `song_embedding` and use those embeddings for recommendations.

The tuned neural network architecture is:

| Layer | Purpose |
|---|---|
| Input layer, 15 features | Receives normalized Spotify feature vector |
| Dense, 512 ReLU | Learns first high-level feature combinations |
| Batch normalization | Stabilizes training |
| Dropout, 0.20 | Reduces overfitting |
| Dense, 256 ReLU | Learns more compressed patterns |
| Batch normalization | Stabilizes training |
| Dropout, 0.15 | Reduces overfitting |
| Dense, 128 ReLU | Learns higher-level song relationships |
| Batch normalization | Stabilizes training |
| Dropout, 0.10 | Reduces overfitting |
| Dense, 64 ReLU | Compresses representation |
| Dense, 48 ReLU, `song_embedding` | Final learned song embedding |
| Dense, 14 softmax | Predicts genre group during training |

The model uses sparse categorical cross-entropy loss and the Adam optimizer with a learning rate of 0.0007. Training can run for up to 40 epochs with a batch size of 128. Early stopping monitors validation accuracy and restores the best weights, while `ReduceLROnPlateau` lowers the learning rate when validation loss stops improving.

### Recommendation Logic

After training, the model generates a 48-dimensional embedding for each song. The recommendation system compares a source song against all other songs using cosine similarity. We also blend in raw audio-feature similarity, popularity similarity, and a small genre-group bonus. The default scoring formula is:

| Signal | Weight |
|---|---:|
| Neural embedding similarity | 0.45 |
| Raw feature similarity | 0.30 |
| Popularity similarity | 0.20 |
| Matching genre group bonus | 0.05 |

This blended score was useful because embedding similarity captures learned musical patterns, while raw feature similarity and popularity help keep recommendations close to the source song in more interpretable ways.

### Baseline Models

We compared our learned embedding approach against a raw-feature cosine similarity baseline. The baseline recommends songs by directly comparing the scaled Spotify feature vectors without using the neural network. This was a useful baseline because it tested whether the neural network learned a better representation than the original features alone.

We also built a simpler neural network architecture during development with dense layers of 256, 128, 64, and a 32-dimensional embedding layer. The tuned model added batch normalization, dropout, a deeper architecture, and a 48-dimensional embedding layer to improve stability and generalization.

## 4. Results and Discussion

### Experimental Configurations

We explored multiple configurations:

| Configuration | Description |
|---|---|
| Raw-feature baseline | Cosine similarity directly on scaled Spotify features |
| Baseline neural network | Dense layers with a 32-dimensional embedding |
| Tuned neural network | Deeper network with batch normalization, dropout, class weights, and 48-dimensional embeddings |
| Blended recommender | Final system combining embeddings, raw features, popularity, and genre-group bonus |

The final model used grouped genres rather than the original raw genre labels. This made the model more stable because many original Spotify genre labels are highly related.

### Quantitative Results

We evaluated recommendation quality by checking whether each song's nearest neighbors belonged to the same broad genre group. Evaluation used a random sample of 5,000 songs from the saved embedding artifact.

| Method | Precision@5 | Hit Rate@5 | Precision@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| Raw feature cosine similarity | 0.3721 | 0.7240 | 0.3529 | 0.8340 |
| Learned neural embeddings | 0.4633 | 0.7802 | 0.4519 | 0.8702 |

Precision@5 measures the average percentage of the top 5 recommendations that match the source song's genre group. Hit Rate@5 measures whether at least one of the top 5 recommendations matches the source genre group. The learned embeddings improved Precision@5 by about 0.0912 over raw feature similarity, showing that the neural network learned a more useful representation for recommendation.

Example recommendation for "Blinding Lights" by The Weeknd:

| Rank | Recommendation | Artist | Score | Embedding Similarity | Feature Similarity | Popularity Match |
|---:|---|---|---:|---:|---:|---:|
| 1 | Ghost | Justin Bieber | 0.955 | 0.968 | 0.856 | 0.970 |
| 2 | Dandelions | Ruth B. | 0.937 | 0.972 | 0.716 | 0.990 |
| 3 | THE LONELIEST | Maneskin | 0.937 | 0.925 | 0.906 | 0.940 |
| 4 | Yet To Come | BTS | 0.935 | 0.949 | 0.830 | 0.940 |
| 5 | MIDDLE OF THE NIGHT | Elley Duhe | 0.930 | 0.964 | 0.699 | 0.990 |

These results are reasonable because all five recommendations fall under the `pop_global` genre group, while still varying across pop, indie-pop, and k-pop.

### Performance Analysis

The learned embeddings performed better than the raw-feature baseline. This suggests that training the model on genre groups helped it learn relationships between features that are not obvious from direct cosine similarity alone. For example, two songs may have similar energy and tempo but different popularity or acousticness, while another song may be more similar overall once the model learns how those features combine.

Batch normalization and dropout were important for model stability. Batch normalization helped keep internal activations in consistent ranges, and dropout reduced overfitting by preventing the network from relying too heavily on individual neurons. Class weighting also helped because the genre groups were not equally represented.

The blended recommender gave us more useful final recommendations than embeddings alone because it balanced learned similarity with interpretable constraints. Popularity matching prevented recommendations from drifting too far away from the source song's general audience level, while the genre bonus helped keep results musically related.

### Qualitative Analysis

The model successfully captures broad musical similarity. For popular songs, it tends to recommend tracks with similar genre groups, energy, and popularity. The "Blinding Lights" example shows this well: the system recommended modern pop-adjacent songs with high embedding similarity and high popularity match.

The model still struggles with subjective similarity. Two songs can have similar Spotify audio features but feel different because of lyrics, artist identity, production style, language, or cultural context. The dataset also does not include user preference data, so the system cannot learn individual taste patterns the way a real streaming service could.

## 5. Conclusion

### Key Findings

Our main result is that learned neural embeddings improved recommendation quality compared with direct raw-feature similarity. On a 5,000-song evaluation sample, learned embeddings achieved Precision@5 of 0.4633 and Hit Rate@5 of 0.7802. The raw-feature baseline achieved Precision@5 of 0.3721 and Hit Rate@5 of 0.7240.

This shows that the neural network learned useful song representations from Spotify audio features. The final recommender also became more practical by blending embedding similarity with raw feature similarity, popularity similarity, and a genre-group bonus.

### Limitations

The biggest limitation is that our dataset does not include user listening behavior. Because of that, our system recommends songs based on content similarity, not personal listening history. Real music platforms usually combine both.

Another limitation is that genre labels are imperfect. Some songs belong to multiple genres, and broad genre groups can hide important differences. The model also does not use lyrics, album context, artist similarity, release year, or user feedback. Finally, the recommendations depend on Spotify's audio feature values, which may not fully represent how people experience music.

### Future Work

Future improvements could include adding user feedback, such as likes or skips, so the recommender could personalize results. We could also include lyrics, artist metadata, release year, or album information. Another improvement would be to test different embedding sizes and scoring weights more systematically. We could also add a larger evaluation section with classification accuracy, precision, recall, F1 score, confusion matrices, and user studies where people rate recommendation quality.

## 6. Reflections and Contributions

### Advice for Future Students

Start with a simple baseline before building the neural network. The raw-feature cosine similarity baseline made it easier to measure whether the learned embeddings actually improved the system. Also, spend time on preprocessing. Grouping related genres, scaling features, and handling class imbalance made the project more manageable.

It is also important to keep evaluation connected to the project goal. Since our goal was recommendation, Precision@K and Hit Rate@K were more useful than only reporting classification accuracy. The classifier helped us train embeddings, but the recommender was the final product.

### Team Contributions

**Jackson Horman**

Tasks and Contributions: Preprocessed the data, improved the training setup, weighted the different recommendation signals, evaluated song similarity, and created the frontend for the recommender.

Personal Reflection: [Add Jackson's reflection here.]

**Ovi Joshi**

Tasks and Contributions: Implemented the neural network architecture, trained the model, tuned hyperparameters, and developed recommendation system logic.

Personal Reflection: [Add Ovi's reflection here.]

**Johanna Sanchez**

Tasks and Contributions: Set up the Jupyter notebook, wrote the data-loading function, removed missing values and duplicate songs, integrated preprocessing into the notebook, addressed class imbalance using scikit-learn class weights, and helped define the TensorFlow/Keras neural network architecture.

Personal Reflection: [Add Johanna's reflection here.]

### Use of Generative AI Tools

We used generative AI tools as coding and writing support during the project. AI assistance was used to help debug code, organize the report, clarify model explanations, and improve wording. The core project decisions, dataset choice, preprocessing strategy, model training, implementation, and evaluation were still directed by the team.

Using AI was appropriate for this project because it helped us work through implementation details and explain our methods more clearly. However, we did not use AI as a replacement for understanding the model or interpreting results. In hindsight, AI was most useful when we asked specific questions about code structure, metrics, or wording, and less useful when the project needed decisions based on our own data and goals.

## 7. References

Maharshi Pandya. "Spotify Tracks Dataset." Kaggle. https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data

TensorFlow Developers. "TensorFlow." https://www.tensorflow.org/

Keras Developers. "Keras." https://keras.io/

scikit-learn Developers. "scikit-learn: Machine Learning in Python." https://scikit-learn.org/

pandas Developers. "pandas." https://pandas.pydata.org/

NumPy Developers. "NumPy." https://numpy.org/
