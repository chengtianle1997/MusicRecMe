# MusicRecMe
A Audio and Lyric Content-based Music Recommendation on Echo Nest Dataset and Spotify Million Playlist Dataset (MPD)
## Data collecting tools - data_loader
1. Setting up a MySQL database
2. Read interaction and collect playlists from Echo Nest and MPD
3. Collecting audio from Spotify Web API
4. Fetching lyric from LyricGenius Web API
5. Getting genre tags from LastFM Web API

## Feature extraction tools - feature_extractor

1. Extract audio features by musicnn
2. Extract lyric features with TF-IDF, Glove, and BERT

## Training and Evaluation

1. The model implementation with PyTorch - model.py

2. Training - train.py
3. Evaluation - evaluate.py
4. Log generator - logger.py
5. Real-time visualization - visdom
