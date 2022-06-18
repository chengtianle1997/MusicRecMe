import os
import json
import numpy as np
from tqdm import tqdm

# Path to generated dataset
echo_nest_sub_path = 'dataset/echo_nest/sub_data/'
echo_nest_whole_path = 'dataset/echo_nest/data/'

# Path to audio and lyric features
out_folder = 'dataset/base/'

# cached matrices and dictionaries name
song_dict_name = 'song_dict.npy'
genre_dict_name = 'genre_dict.npy' # for a genre tag list with index for genre matrix
genre_mat_name = 'genre_mat.npy'
meta_mat_name = 'meta_mat.npy'
# other feature mat named as: {}_mat.npy, such as: musicnn_mat.npy

meta_dict = {'key': 0, 'tempo': 1, 'energy': 2, 'valence': 3, 'liveness': 4, 'loudness': 5, \
    'duration_ms': 6, 'speechiness': 7, 'acousticness': 8, 'danceability': 9, 'time_signature': 10, \
    'instrumentalness': 11, 'mode': 12}

audio_feature_list = ['musicnn']
lyric_feature_list = ['tf-idf', 'doc2vec']

class Dataset(object):

    # dataset_root: root for the dataset folder
    # sub [True, False]: using sub dataset for debugging and testing
    # genre [True, False]: genre tag
    # meta [True, False]: audio features provided by Spotify
    # audio [None, 'musicnn', ...]: extracted audio features by specified method
    # lyric [None, 'tf-idf', 'doc2vec', ...]: extracted lyric features by specified method
    def __init__(self, dataset_root='E:/dataset_old', sub=True, \
        genre=False, meta=True, audio='musicnn', lyric=None):
        # Save params
        self.dataset_root = dataset_root
        self.sub = sub
        self.genre = genre
        self.meta = meta
        self.audio = audio
        self.lyric = lyric
        # get paths
        if sub is True:
            self.outdir = self.dataset_root + '/' + echo_nest_sub_path
        else:
            self.outdir = self.dataset_root + '/' + echo_nest_whole_path
        self.train_song_json_url = self.outdir + 'train_song.json'
        self.test_song_json_url = self.outdir + 'test_song.json'
        self.song_json_url = self.outdir + 'song.json'
        self.train_folder = self.outdir + 'train'
        self.valid_folder = self.outdir + 'valid'
        self.test_folder = self.outdir + 'test'
        # load song json
        self.song_json = self.load_json(self.song_json_url)
        if self.song_json is None:
            exit(0)
        # load song dict
        self.song_dict = self.load_song_dict()
        if self.song_dict is None:
            exit(0)
        # load genre dict and matrix
        if self.genre is True:
            self.genre_dict, self.genre_mat = self.load_genre_dict()
        # load meta matrix
        if self.meta is True:
            self.meta_mat = self.load_meta_dict()
        # load audio matrix
        if self.audio is not None:
            self.audio_mat = self.load_audio_features(merge='avg')
            if self.audio_mat is None:
                exit(0)

    # load json file as a dictionary
    def load_json(self, json_url):
        try:
            jsonf = open(json_url, 'r', encoding='utf-8')
        except:
            print("Fatal: Cannot open json file {}".format(json_url))
            return None
        json_cont = json.load(jsonf)
        return json_cont

    # load song json as a dictionary
    def load_song_dict(self):
        song_dict_path = self.outdir + song_dict_name
        # check if there is song dictionary cache exists
        if os.path.exists(song_dict_path):
            song_dict = np.load(song_dict_path, allow_pickle='TRUE').item()
            print("Load existed song dictionary cache successfully!")
            return song_dict
        # load song dict from song json file
        print("No song dictionary cache exists, load it from {}...".format(self.song_json_url))
        song_dict = {}
        for track_id in tqdm(self.song_json.keys()):
            song_dict[track_id] = len(song_dict)
        # save song dict as cache file
        np.save(song_dict_path, song_dict)
        return song_dict

    # load genre tag dict
    # tag: 'genre_top' or 'genre_raw'
    def load_genre_dict(self, tag='genre_top'):
        genre_dict_path = self.outdir + genre_dict_name
        genre_mat_path = self.outdir + genre_mat_name
        genre_dict = None
        genre_mat = None
        # check if there are genre dictionary and matrix cache exist
        if os.path.exists(genre_dict_path):
            genre_dict = np.load(genre_dict_path, allow_pickle='TRUE').item()
            print("Load existed genre dictionary cache successfully!")
        if os.path.exists(genre_mat_path):
            genre_mat = np.load(genre_mat_path)
            print("Load existed genre matrix cache successfully!")
        if genre_dict is not None and genre_mat is not None:
            return genre_dict, genre_mat
        # load genre dictionary and matrix from song json and dictionary
        print("No genre cache file exists, load it from song json...")
        genre_dict = {}
        for track_id in tqdm(self.song_dict.keys()):
            tags = json.loads(self.song_json[track_id][tag])
            if tags is None:
                continue
            for genre_tag in tags.keys():
                if not genre_tag in genre_dict.keys():
                    genre_dict[genre_tag] = len(genre_dict)
        # load genre matrix
        print("Load genre matrix...")
        genre_mat = np.zeros((len(self.song_dict), len(genre_dict)), dtype=np.float32)
        for track_id in tqdm(self.song_dict.keys()):
            tags = json.loads(self.song_json[track_id][tag])
            if tags is None:
                continue
            for genre_tag in tags.keys():
                genre_mat[self.song_dict[track_id]][genre_dict[genre_tag]] = float(tags[genre_tag]) / 100
        # save genre dictionary and matrix as cache file
        np.save(genre_dict_path, genre_dict)
        np.save(genre_mat_path, genre_mat)
        return genre_dict, genre_mat
            
    # load meta info: audio features provided by Spotify
    def load_meta_dict(self):
        meta_mat_path = self.outdir + meta_mat_name
        # check if there is meta matrix cache exists
        if os.path.exists(meta_mat_path):
            meta_mat = np.load(meta_mat_path)
            print("Load existed meta info matrix cache successfully!")
            return meta_mat
        # load meta matrix from song json and dictionary
        print("No meta cache file exists, load it from song json...")
        meta_mat = np.zeros((len(self.song_dict), len(meta_dict)))
        for track_id in tqdm(self.song_dict.keys()):
            meta_info = json.loads(self.song_json[track_id]['audio_features'])
            if meta_info is None:
                continue
            for meta_key in meta_dict.keys():
                meta_mat[self.song_dict[track_id]][meta_dict[meta_key]] = meta_info[meta_key]
        # standardize the features
        meta_mat = (meta_mat - meta_mat.min(axis=0)) / meta_mat.ptp(axis=0)
        # save meta matrix as cache file
        np.save(meta_mat_path, meta_mat)
        return meta_mat

    # load audio features extracted from pre-trained model, e.g: musicnn
    # merge (method to merge sequence): 'avg': average pooling, 'max': max pooling
    def load_audio_features(self, merge='avg'):
        audio_mat_path = self.outdir + '{}_{}_mat.npy'.format(self.audio, merge)
        # check if there is cached matrix
        if os.path.exists(audio_mat_path):
            audio_mat = np.load(audio_mat_path)
            print("Load existed audio feature matrix cache successfully!")
            return audio_mat
        # load audio feature matrix from audio feature folder
        print("No audio feature cache file exists, load it from extracted features...")
        audio_feature_root = self.dataset_root + '/' + out_folder + \
            '/{}_features'.format(self.audio)
        # load one matrix to get the shape
        if not os.path.exists(audio_feature_root):
            print("Fail to find extracted audio features: {} not exists".format(audio_feature_root))
            return None
        audio_feature_files = os.listdir(audio_feature_root)
        if len(audio_feature_files) < 1:
            print("No extracted audio feature file exists in {}".format(audio_feature_root))
            return None
        audio_feature_sample = np.load(audio_feature_root + '/' + audio_feature_files[0])
        audio_feature_shape = audio_feature_sample.shape
        if self.audio == 'musicnn':
            audio_mat = np.zeros((len(self.song_dict), audio_feature_shape[1]))
            unfound_track_counter = 0
            for track_id in tqdm(self.song_dict.keys()):
                ori_track_id = track_id
                # convert track_id to file name
                if not track_id.find("spotify") == -1:
                    track_id = track_id.split(":")[2]
                track_id += '.npy'
                # make sure the file exists
                if track_id in audio_feature_files:
                    track_mat = np.load(audio_feature_root + '/' + track_id)
                    # sequence merging
                    if merge == 'avg':
                        track_mat = track_mat.mean(axis=0)
                    if merge == 'max':
                        track_mat = track_mat.max(axis=0)
                    # save to audio matrix
                    audio_mat[self.song_dict[ori_track_id]][:] = track_mat
                else:
                    unfound_track_counter += 1
                    print("[Audio feature]Unfound track No.{}: {}".format(unfound_track_counter, ori_track_id))
                    continue
        # save audio matrix to cache file
        np.save(audio_mat_path, audio_mat)
        return audio_mat
        


if __name__ == '__main__':
    dataset = Dataset()