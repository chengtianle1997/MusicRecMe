# spotifyApi.py
# Spotify api to download music audio and meta-information

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from fuzzywuzzy import fuzz
import os
import api_key
import datetime

# API client id and secret (from spotify dashboard)
keys = api_key.read_spotify_keys()
client_id = keys['client_id']
client_secret = keys['client_secret']

# Path configuration
music_path = 'dataset/base/music'

# The format of return dictionary
# res_dict =
# {
# "track_id": track id,
# "track_name": track name (song title),
# "album_id": album id,
# "album_name": album name,
# "artist_id": artist id,
# "artist_name": artist name,
# "release_date": release date of the track (album in fact),
# "audio_features": audio feature dictionary,
# "preview_url": preview url to download the audio file
# }

# check if the song is a new song
def is_valid(time, thres='2012-01-01'):
    if time is None:
        return False
    time_date = datetime.datetime.strptime(time, '%Y-%m-%d').date()
    thres_date = datetime.datetime.strptime(thres, '%Y-%m-%d').date()
    return time_date < thres_date


class Spotify(object):
    def __init__(self, dataset_root=''):
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(\
            client_id=client_id, client_secret=client_secret))
        # create path to save music audio files
        self.music_path = dataset_root + '/' + music_path
        if not os.path.exists(self.music_path):
            os.makedirs(self.music_path)

    # Search with track name (for echo nest dataset)
    # Input: 
    # track_name: track name (song title)
    # artist_name: corresponding artist name (can be None)
    # Return:
    # success: res_dict
    # fail: None
    def search_track_name(self, track_name, artist=None):
        # make sure track_name and artist are all string
        track_name = str(track_name)
        artist = str(artist)
        # Init a empty res dictionary
        res_dict = {
            "track_id": None,
            "track_name": track_name,
            "album_id": None,
            "album_name": None,
            "artist_id": None,
            "artist_name": artist,
            "release_date": None,
            "audio": 0,
            "preview_url": None,
            "audio_features": None
        }
        if track_name is None:
            return res_dict
        query_cmd = 'track:{}'.format(track_name)
        try:
            res = self.sp.search(q=query_cmd, type='track', limit=5)
        except:
            return res_dict
        # check if the search result is not empty
        if len(res['tracks']['items']) == 0:
            return res_dict
        # ratio score list
        ratio_list = []
        for idx, track in enumerate(res['tracks']['items']):
            # compare the search result with the given artist name
            if artist is not None:
                # if there is more than one artist, choose the most similar one
                ratio_list.append(max([fuzz.ratio(artist, x['name']) for x in track['artists']]))
        # if the artist name is not given, choose the first one
        max_id = 0
        # if the artist name is given, choose the most similar name
        if artist is not None:
            max_id = ratio_list.index(max(ratio_list))
        # take the track id, release date, preview url 
        track_id = res['tracks']['items'][max_id]['uri']
        album_id = res['tracks']['items'][max_id]['album']['uri']
        album_name = res['tracks']['items'][max_id]['album']['name']
        artist_id = res['tracks']['items'][max_id]['artists'][0]['uri']
        artist_name = res['tracks']['items'][max_id]['artists'][0]['name']
        release_date = res['tracks']['items'][max_id]['album']['release_date']
        preview_url = res['tracks']['items'][max_id]['preview_url']
        is_audio_exist = 1 if preview_url is not None else 0 
        
        # check release date
        ymd = release_date.split('-')
        if len(ymd) == 2:
            release_date += '-01'
        elif len(ymd) == 1:
            release_date += '-01-01'

        # get audio features
        audio_feature_dict = self.sp.audio_features(track_id)[0]
        # download the 30s preview audio file
        if preview_url is not None:
            mp3_cont = requests.get(preview_url, allow_redirects=True)
            mp3_path = "{}/{}.mp3".format(self.music_path, track_id.split(":")[2])
            open(mp3_path, 'wb').write(mp3_cont.content)
        # set the dict
        res_dict = {
            "track_id": track_id,
            "track_name": track_name,
            "album_id": album_id,
            "album_name": album_name,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "release_date": release_date,
            "audio": is_audio_exist,
            "preview_url": preview_url,
            "audio_features": audio_feature_dict
        }
        return res_dict

    # Get track id with song title and artist name
    def search_track_name_for_id(self, track_name, artist=None):
        # make sure track_name and artist are all string
        track_name = str(track_name)
        artist = str(artist)
        # Init a empty res dictionary
        res_dict = {
            "track_id": None,
            "track_name": track_name,
            "album_id": None,
            "album_name": None,
            "artist_id": None,
            "artist_name": artist,
            "release_date": None,
            "audio": 0,
            "preview_url": None,
            "audio_features": None
        }
        if track_name is None:
            return res_dict
        if artist is not None:
            query_cmd = 'track:{}, artist:{}'.format(track_name, artist)
        else:
            query_cmd = 'track:{}'.format(track_name)
        try:
            res = self.sp.search(q=query_cmd, type='track', limit=5)
        except:
            return res_dict
        # check if the search result is not empty
        if len(res['tracks']['items']) == 0:
            return res_dict
        # ratio score list
        # ratio_list = []
        # for idx, track in enumerate(res['tracks']['items']):
        #     # compare the search result with the given artist name
        #     if artist is not None:
        #         # if there is more than one artist, choose the most similar one
        #         ratio_list.append(max([fuzz.ratio(artist, x['name']) for x in track['artists']]))

        max_id = 0
        # select the track with proper release date (before 2012-01-01)
        for idx in range(0, len(res['tracks']['items'])):
            release_date_i = res['tracks']['items'][idx]['album']['release_date']
            # check release date
            ymd = release_date_i.split('-')
            if len(ymd) == 2:
                release_date_i += '-01'
            elif len(ymd) == 1:
                release_date_i += '-01-01'
            if is_valid(release_date_i):
                max_id = idx
                break
        
        # if the artist name is given, choose the most similar name
        # if artist is not None:
        #     max_id = ratio_list.index(max(ratio_list))
        # take the track id, release date, preview url 
        track_id = res['tracks']['items'][max_id]['uri']
        album_id = res['tracks']['items'][max_id]['album']['uri']
        album_name = res['tracks']['items'][max_id]['album']['name']
        artist_id = res['tracks']['items'][max_id]['artists'][0]['uri']
        artist_name = res['tracks']['items'][max_id]['artists'][0]['name']
        release_date = res['tracks']['items'][max_id]['album']['release_date']
        preview_url = res['tracks']['items'][max_id]['preview_url']
        is_audio_exist = 1 if preview_url is not None else 0 
        
        # check release date
        ymd = release_date.split('-')
        if len(ymd) == 2:
            release_date += '-01'
        elif len(ymd) == 1:
            release_date += '-01-01'

        # set the dict
        res_dict = {
            "track_id": track_id,
            "track_name": track_name,
            "album_id": album_id,
            "album_name": album_name,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "release_date": release_date,
            "audio": is_audio_exist,
            "preview_url": preview_url,
            "audio_features": None
        }
        
        return res_dict
    
    # download audio features and 30s preview audio
    # Input:
    # res_dict: from search_track_name or search_track_name_for_id
    # Return:
    # success: res_dict
    # fail: None
    def download_audio(self, res_dict):
        track_id = res_dict['track_id']
        # get audio features
        audio_feature_dict = self.sp.audio_features(track_id)[0]
        res_dict['audio_features'] = audio_feature_dict
        # download the 30s preview audio file
        preview_url = res_dict['preview_url']
        if preview_url is not None:
            mp3_cont = requests.get(preview_url, allow_redirects=True)
            mp3_path = "{}/{}.mp3".format(self.music_path, track_id.split(":")[2])
            open(mp3_path, 'wb').write(mp3_cont.content)
        return res_dict

    # search with track id (for spotify million playlist dataset)
    # Input:
    # track_id: spotify format track id. e.g: 'spotify:track:23EOmJivOZ88WJPUbIPjh6'
    # Return:
    # success: res_dict
    # fail: None
    def search_track_id(self, track_id):
        # search track with track id
        if track_id is None:
            return None
        try:
            res = self.sp.track(track_id)
        except:
            return None
        # fill in the result dictionary
        track_name = res['name']
        album_id = res['album']['uri']
        album_name = res['album']['name']
        artist_id = res['artists'][0]['uri']
        artist_name = res['artists'][0]['name']
        release_date = res['album']['release_date']
        preview_url = res['preview_url']
        is_audio_exist = 1 if preview_url is not None else 0 
        # check release date
        ymd = release_date.split('-')
        if len(ymd) == 2:
            release_date += '-01'
        elif len(ymd) == 1:
            release_date += '-01-01'
        # get audio features
        audio_feature_dict = self.sp.audio_features(track_id)[0]
        # download the 30s preview audio file
        if preview_url is not None:
            mp3_cont = requests.get(preview_url, allow_redirects=True)
            mp3_path = "{}/{}.mp3".format(self.music_path, track_id.split(":")[2])
            open(mp3_path, 'wb').write(mp3_cont.content)
        # set the res_dict
        res_dict = {
            "track_id": track_id,
            "track_name": track_name,
            "album_id": album_id,
            "album_name": album_name,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "release_date": release_date,
            "audio": is_audio_exist,
            "preview_url": preview_url,
            "audio_features": audio_feature_dict
        }
        
        return res_dict