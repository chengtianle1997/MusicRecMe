import json

def read_spotify_keys(file_path='api_key.json', idx=0):
    try:
        f = open(file_path)
        key_dict = json.load(f)
        return key_dict['spotify'][idx]
    except:
        print("Cannot find {}".format(file_path))

def read_lastfm_keys(file_path='api_key.json', idx=0):
    try:
        f = open(file_path)
        key_dict = json.load(f)
        return key_dict['lastfm'][idx]
    except:
        print("Cannot find {}".format(file_path))

def read_lyricgenius_keys(file_path='api_key.json', idx=0):
    try:
        f = open(file_path)
        key_dict = json.load(f)
        return key_dict['lyricgenius'][idx]
    except:
        print("Cannot find {}".format(file_path))

def read_mysql_keys(file_path='api_key.json', idx=0):
    try:
        f = open(file_path)
        key_dict = json.load(f)
        return key_dict['mysql'][idx]['password']
    except:
        print("Cannot find {}".format(file_path))