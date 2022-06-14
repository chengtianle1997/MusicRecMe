# lastFM.py 
# Get genres tags from LastFM

import pylast
import api_key

# API info
keys = api_key.read_lastfm_keys()
API_KEY = keys["API_KEY"]
API_SECRET = keys["API_SECRET"]
username = keys["username"]
password_hash = keys["password_hash"]

# lastFM tag list path
lastFM_tag_txt = 'dataset/lastfm_unique_tags.txt'

# number of tags
MAX_TAG_NUM = 200

class lastFM(object):
    def __init__(self, dataset_root='E:'):
        f_path = dataset_root + '/' + lastFM_tag_txt
        # Get the tag list
        try:
            f = open(f_path)
        except:
            print("No file named {} found!".format(f_path))
            exit
        tag_list = []
        for data in f:
            tag_list.append(data.split('\t')[0])
            if len(tag_list) > MAX_TAG_NUM - 1:
                break
        # Convert tag list to tag set for calculating intersection
        self.tag_set = set(tag_list)
        # Init LastFM API
        self.network = pylast.LastFMNetwork(
            api_key=API_KEY,
            api_secret=API_SECRET,
            username=username,
            password_hash=password_hash,
        )
    
    # Get tag from lastFM by artist name and song title
    def get_tag(self, artist=None, song_title=None):
        # Check if artist and song title is valid
        if artist is None or song_title is None:
            return {}, {}
        # Get track
        try:
            track = self.network.get_track(artist, song_title)
            tags = track.get_top_tags()
        except:
            print("LastFM: Cannot find track: {}, artist: {}".format(song_title, artist))
            return {}, {}
        # Return two dictionary: 
        # res (all tags in tag set) and res_sorted (Top 5 tags only)
        res = {}
        tag_n = {tag.item.name: tag.weight for tag in tags}
        # Calculate intersection of track tags and tag_set
        tag_n_set = set(tag_n.keys())
        tag_inter = tag_n_set & self.tag_set
        for tag in tag_inter:
            res[tag] = int(tag_n[tag])
        # Sort the tag by weight to find Top 5 tags
        res_sorted = sorted(res.items(), key=lambda x: x[1], reverse=True)
        res_sorted = res_sorted[:5]
        # convert res_sorted from tuple to dict
        res_sorted = {x[0]: x[1] for x in res_sorted}
        return res, res_sorted

        

        
