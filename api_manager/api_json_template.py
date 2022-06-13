# api_json_template.py
# Use this file to generate api_key.json file

import json
import pylast

key_dict = {}

# Sign up and create an application here for Spotify API: 
# https://developer.spotify.com/dashboard/login
# Doc: https://spotipy.readthedocs.io/en/2.19.0/#examples

key_dict['spotify'] = [
    {
        'client_id': "-----client id 1------",
        'client_secret': "-----client secret 1------"
    },
    {
        'client_id': "-----client id 2------",
        'client_secret': "-----client secret 2------"
    }
]

# Sign up and create an application here for LastFM API:
# https://www.last.fm/api/account/create
# Doc: https://www.last.fm/api

key_dict['lastfm'] = [
    {
        "API_KEY": "-----API KEY------",
        "API_SECRET": "-----API SECRET------",
        "username": "----user name, not email address----",
        "password_hash": pylast.md5("-----your password------")
    }
]

# Sign up and create an application here for LyricGenius API:
# https://genius.com/api-clients
# Doc: https://docs.genius.com/

key_dict['lyricgenius'] = [
    {
        "client_access_token": "----lyric genius client access token----"
    }
]

key_dict['mysql'] = [
    {
        "password": "-----MySQL password-----"
    }
]

with open('api_key.json', 'w') as f:
    json.dump(key_dict, f)