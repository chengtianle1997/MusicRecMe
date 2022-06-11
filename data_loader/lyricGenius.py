# lyricGenius.py
# Download full-text lyric from lyric Genius API

import lyricsgenius as lg
import os
import api_key

# Client access token for lyricsGenius API
keys = api_key.read_lyricgenius_keys()
client_access_token = keys["client_access_token"]

# Path to save lyric text file
lyric_path = 'dataset/base/lyric_raw'

class LyricGenius(object):
    def __init__(self, dataset_root=''):
        self.genius = lg.Genius(client_access_token, skip_non_songs=True, \
            excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
        # check if lyric path exists
        self.lyric_path = dataset_root + '/' + lyric_path
        if not os.path.exists(self.lyric_path):
            os.makedirs(self.lyric_path)
    
    # Search lyric by track name and artist name
    # Input: 
    # track_id: txt file name (to save the lyric text)
    # track_name, artist_name: to search the lyric for specific track
    # Return:
    # success: True, save the lyric to file named track_id.txt
    def search_track(self, track_id, track_name, artist_name):
        try:
            song = self.genius.search_song(str(track_name), artist=str(artist_name))
        except:
            return False

        if song is None:
            return False

        # check if it is a spotify id or a echo nest id
        if not track_id.find("spotify") == -1:
            track_id = track_id.split(":")[2]
        lyric_file_path = "{}/{}.txt".format(self.lyric_path, track_id)
        with open(lyric_file_path, 'w', encoding='utf-8') as f:
            f.write(song.lyrics)
            return True
        return False # for saving failure


