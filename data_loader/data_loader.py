import os
import json
import numpy as np
from tqdm import tqdm
import pickle
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sklearn.decomposition import PCA

# set a random seed
random.seed(2022)

# Path to generated dataset
echo_nest_sub_path = 'dataset/echo_nest/sub_data/'
echo_nest_whole_path = 'dataset/echo_nest/data/'

# Path to audio and lyric features
out_folder = 'dataset/base/'

# cached matrices and dictionaries name
song_dict_name = 'song_dict.npy'
song_old_new_dict_name = 'song_old_new_dict.npy'
genre_dict_name = 'genre_dict.npy' # for a genre tag list with index for genre matrix
genre_mat_name = 'genre_mat.npy'
meta_mat_name = 'meta_mat.npy'
one_hot_mat_name = 'one_hot_mat.npy'
# other feature mat named as: {}_mat.npy, such as: musicnn_mat.npy

meta_dict = {'key': 0, 'tempo': 1, 'energy': 2, 'valence': 3, 'liveness': 4, 'loudness': 5, \
    'duration_ms': 6, 'speechiness': 7, 'acousticness': 8, 'danceability': 9, 'time_signature': 10, \
    'instrumentalness': 11, 'mode': 12}

audio_feature_list = ['musicnn']
lyric_feature_list = ['tf_idf', 'doc2vec']

class Dataset(object):

    # dataset_root: root for the dataset folder
    # sub [True, False]: using sub dataset for debugging and testing
    # genre [True, False]: genre tag
    # meta [True, False]: audio features provided by Spotify
    # audio [None, 'musicnn', ...]: extracted audio features by specified method
    # lyric [None, 'tf-idf', 'doc2vec', ...]: extracted lyric features by specified method
    def __init__(self, dataset_root='E:', sub=True, \
        genre=False, meta=True, audio='musicnn', lyric=None, outdir=None, dim_clip='pca',\
        dim_list=[64, 0, 128, 128]):
        # Save params
        self.dataset_root = dataset_root
        self.sub = sub
        self.genre = genre
        self.meta = meta
        self.audio = audio
        self.lyric = lyric
        self.dim_clip = dim_clip
        self.dim_list = dim_list
        # get paths
        if sub is True:
            self.indir = self.dataset_root + '/' + echo_nest_sub_path
        else:
            self.indir = self.dataset_root + '/' + echo_nest_whole_path
        if outdir is not None:
            self.outdir = outdir + '/'
        else:
            self.outdir = self.indir
        self.train_song_json_url = self.indir + 'train_song.json'
        self.test_song_json_url = self.indir + 'test_song.json'
        self.song_json_url = self.indir + 'song.json'
        self.train_folder = self.indir + 'train'
        self.valid_folder = self.indir + 'valid'
        self.test_folder = self.indir + 'test'
        # load song json
        self.song_json = self.load_json(self.song_json_url)
        if self.song_json is None:
            exit(0)
        # load song dict
        self.song_dict, self.song_old_new_dict = self.load_song_dict()
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
        # load lyric matrix
        if self.lyric is not None:
            self.lyric_mat = self.load_lyric_features()
            if self.lyric_mat is None:
                exit(0)
        # load song matrix
        self.song_mat = self.get_song_mat()
        # load train and test song matrix
        self.train_song_dict, self.train_song_mat = \
            self.get_train_test_song_mat(self.train_song_json_url, set_tag='train')
        self.test_song_dict, self.test_song_mat = \
            self.get_train_test_song_mat(self.test_song_json_url, set_tag='test')
        # load training set
        self.train_mat = self.load_data(self.train_folder, set_tag='train')
        self.valid_mat = self.load_data(self.valid_folder, set_tag='valid')
        self.test_mat = self.load_data(self.test_folder, set_tag='test')
        if self.train_mat is None or self.valid_mat is None or self.test_mat is None:
            exit(0)
        self.x_len_max, self.y_len_max = self.get_max_seq_len()
        

    # Public APIs -----------------------------------------------------------------
    # get the dimension of all embeddings
    # Return:
    # success: 
    # music_dim: the dimension of the whole music embedding
    # [genre_dim, meta_dim, audio_dim, lyric_dim]: the list of different kinds of embeddings
    def get_dim(self):
        genre_dim = 0
        meta_dim = 0
        audio_dim = 0
        lyric_dim = 0
        if self.genre is True:
            genre_dim = self.genre_mat.shape[1]
            if self.dim_list[0] > 0:
                genre_dim = self.dim_list[0]
        if self.meta is True:
            meta_dim = self.meta_mat.shape[1]
            if self.dim_list[1] > 0:
                meta_dim = self.dim_list[1]
        if self.audio is not None:
            audio_dim = self.audio_mat.shape[1]
            if self.dim_list[2] > 0:
                audio_dim = self.dim_list[2]
        if self.lyric is not None:
            lyric_dim = self.lyric_mat.shape[1]
            if self.dim_list[3] > 0:
                lyric_dim = self.dim_list[3]
        music_dim = genre_dim + meta_dim + audio_dim + lyric_dim
        # dimension check
        if not music_dim == self.song_mat.shape[1]:
            print("[Dataset Matrix Dimension] Dimension alignment failed!")
            exit(0)
        return music_dim, [genre_dim, meta_dim, audio_dim, lyric_dim]

    # get train, valid and test data in track id list format
    # set_tag = ['train', 'valid', 'test']
    # Return: [x_len, y_len, x_mat, y_mat]
    # x_len: list of song numbers in x
    # y_len: list of song numbers in y
    # x_mat: list of songs track ids in x
    # y_mat: list of songs track ids in y
    def get_data(self, shuffle=True, set_tag='train', neg_samp=False, neg_pos_ratio=10):
        # training set
        if set_tag == 'train':
            if shuffle is True:
                random.shuffle(self.train_mat)
            x_len = [len(data['x']) for data in self.train_mat]
            y_len = [len(data['y']) for data in self.train_mat]
            x_mat = [data['x'] for data in self.train_mat]
            y_mat = [data['y'] for data in self.train_mat]     
        # valid set
        elif set_tag == 'valid':
            x_len = [len(data['x']) for data in self.valid_mat]
            y_len = [len(data['y']) for data in self.valid_mat]
            x_mat = [data['x'] for data in self.valid_mat]
            y_mat = [data['y'] for data in self.valid_mat]
        # test set
        elif set_tag == 'test':
            x_len = [len(data['x']) for data in self.test_mat]
            y_len = [len(data['y']) for data in self.test_mat]
            x_mat = [data['x'] for data in self.test_mat]
            y_mat = [data['y'] for data in self.test_mat]
        # negative sampling
        if neg_samp is True:
            y_neg_len, y_neg_mat = self.get_neg_data(x_mat, y_mat, k=neg_pos_ratio)
            return [x_len, y_len, x_mat, y_mat, y_neg_len, y_neg_mat]
        else:
            return [x_len, y_len, x_mat, y_mat]
    
    # get negative samples randomly from other songs
    def get_neg_data(self, x_mat, y_mat, k=10):
        all_set = set(self.song_dict.keys())
        y_neg_mat = []
        y_neg_len = []
        for idx in range(len(x_mat)):
            in_set = set(x_mat[idx]) | set(y_mat[idx])
            other_set = list(all_set - in_set)
            # random sampling
            y_neg_mat.append(random.sample(other_set, min(k * len(y_mat[idx]), 2000)))
            y_neg_len.append(min(k * len(y_mat[idx]), 2000))
        return y_neg_len, y_neg_mat

    # get dict and matrix, set_tag = [None (for all songs), 'train', 'test']
    # Return:
    # song_dict: song dictionary, where key = track id, value = index for song matrix
    # song_mat: song matrix, contains song embeddings
    def get_dict_mat(self, set_tag=None):
        if set_tag is None:
            return self.song_dict, self.song_mat
        elif set_tag == 'train':
            return self.train_song_dict, self.train_song_mat
        elif set_tag == 'test':
            return self.test_song_dict, self.test_song_mat
        else:
            print("[Get Dict and Mat Err]: no {} set_tag found!".format(set_tag))

    # get batched one-hot format data
    def get_onehot_data(self, len_mat_res, batch_size=50, mode='train'):
        x_len, y_len, x_mat, y_mat = len_mat_res[0], len_mat_res[1], len_mat_res[2], len_mat_res[3]
        batch_num = int(len(x_len) / batch_size)
        x_batch_list = []
        x_inv_batch_list = []
        y_batch_list = []
        if mode == 'train' or mode == 'valid':
            song_dict = self.train_song_dict
        else:
            song_dict = self.test_song_dict
        song_dict_dim = len(song_dict)
        for batch_idx in range(batch_num):
            x_batch = np.zeros((batch_size, song_dict_dim))
            x_inv_batch = np.ones((batch_size, song_dict_dim))
            y_batch = np.zeros((batch_size, song_dict_dim))
            for row_idx in range(batch_size):
                x_index = [song_dict[x] for x in x_mat[batch_idx * batch_size + row_idx]]
                y_index = [song_dict[y] for y in y_mat[batch_idx * batch_size + row_idx]]
                x_batch[row_idx, x_index] = 1
                x_inv_batch[row_idx, x_index] = 0
                y_batch[row_idx, y_index] = 1
            x_batch_list.append(torch.tensor(x_batch))
            x_inv_batch_list.append(torch.tensor(x_inv_batch))
            y_batch_list.append(torch.tensor(y_batch))
        return x_batch_list, x_inv_batch_list, y_batch_list


    # get batched train, valid, and test data in tensor format (music embedding inserted)
    # len_mat_res: return list from get_data() method
    # batch_size: batch size
    # fix_length: True = all batched data are in max_length, same sequence length among batches
    #             False = only data in one batch are in the same length, various sequence length among batches
    def get_batched_data(self, len_mat_res, batch_size=50, fix_length=False, neg_samp=False):
        if neg_samp is True:
            x_len, y_len, x_mat, y_mat, y_neg_len, y_neg_mat = len_mat_res[0], len_mat_res[1], len_mat_res[2], len_mat_res[3], len_mat_res[4], len_mat_res[5]
        else:
            x_len, y_len, x_mat, y_mat = len_mat_res[0], len_mat_res[1], len_mat_res[2], len_mat_res[3]
        batch_num = int(len(x_len) / batch_size)
        x_len_list = []
        y_len_list = []
        y_neg_len_list = []
        x_mat_tensor_list = []
        y_mat_tensor_list = []
        # convert mat to list of tensors -> [sample_num, playlist_length, music_embed_dim]
        x_mat = [torch.tensor(np.array([self.song_mat[self.song_dict[track_id]] for track_id in x]), dtype=torch.float32) for x in x_mat]
        y_mat = [torch.tensor(np.array([self.song_mat[self.song_dict[track_id]] for track_id in y]), dtype=torch.float32) for y in y_mat]
        if neg_samp is True:
            y_neg_mat = [torch.tensor(np.array([self.song_mat[self.song_dict[track_id]] for track_id in y]), dtype=torch.float32) for y in y_neg_mat]
            y_neg_mat_tensor_list = []
        # iterate batches
        for batch_idx in range(batch_num):
            x_len_list.append(x_len[batch_idx * batch_size: (batch_idx + 1) * batch_size])
            y_len_list.append(y_len[batch_idx * batch_size: (batch_idx + 1) * batch_size])
            if neg_samp is True:
                y_neg_len_list.append(y_neg_len[batch_idx * batch_size: (batch_idx + 1) * batch_size])
            if fix_length is True:
                # pad the first sequence to disired length
                x_mat[batch_idx * batch_size] = nn.ConstantPad2d\
                    ((0, 0, self.x_len_max - x_mat[batch_idx * batch_size].shape[0], 0), 0)(x_mat[batch_idx * batch_size])
                y_mat[batch_idx * batch_size] = nn.ConstantPad2d\
                    ((0, 0, self.y_len_max - y_mat[batch_idx * batch_size].shape[0], 0), 0)(y_mat[batch_idx * batch_size])
                if neg_samp is True:
                    y_neg_mat[batch_idx * batch_size] = nn.ConstantPad2d\
                    ((0, 0, self.y_len_max - y_neg_mat[batch_idx * batch_size].shape[0], 0), 0)(y_neg_mat[batch_idx * batch_size])
            # padding
            x_mat_padded_idx = pad_sequence(x_mat[batch_idx * batch_size: (batch_idx + 1) * batch_size], batch_first=True)
            x_mat_tensor_list.append(x_mat_padded_idx) # [batch_size, padded_seq_len, music_embed_dim]
            y_mat_padded_idx = pad_sequence(y_mat[batch_idx * batch_size: (batch_idx + 1) * batch_size], batch_first=True)
            y_mat_tensor_list.append(y_mat_padded_idx) # [batch_size, padded_seq_len, music_embed_dim]
            if neg_samp is True:
                y_neg_mat_padded_idx = pad_sequence(y_neg_mat[batch_idx * batch_size: (batch_idx + 1) * batch_size], batch_first=True)
                y_neg_mat_tensor_list.append(y_neg_mat_padded_idx) # [batch_size, padded_seq_len, music_embed_dim]
        if neg_samp is True:
            return x_len_list, y_len_list, x_mat_tensor_list, y_mat_tensor_list, y_neg_len_list, y_neg_mat_tensor_list
        else:
            return x_len_list, y_len_list, x_mat_tensor_list, y_mat_tensor_list


    # get the maximum sequence length among train, valid and test data
    def get_max_seq_len(self):
        x_train_len = [len(data['x']) for data in self.train_mat]
        y_train_len = [len(data['y']) for data in self.train_mat]
        x_valid_len = [len(data['x']) for data in self.valid_mat]
        y_valid_len = [len(data['y']) for data in self.valid_mat]
        x_test_len = [len(data['x']) for data in self.test_mat]
        y_test_len = [len(data['y']) for data in self.test_mat]
        x_len_max = max(max(x_train_len), max(x_valid_len), max(x_test_len))
        y_len_max = max(max(y_train_len), max(y_valid_len), max(y_test_len))
        return x_len_max, y_len_max
        
    # ------------------------------------------------------------------------------


    # load json file as a dictionary
    def load_json(self, json_url):
        try:
            jsonf = open(json_url, 'r', encoding='utf-8')
        except:
            print("Fatal: Cannot open json file {}".format(json_url))
            return None
        json_cont = json.load(jsonf)
        return json_cont

    # load train, valid and test set:
    def load_data(self, data_folder, set_tag='train'):
        data_mat_path = self.outdir + "{}_mat.pkl".format(set_tag)
        # check if cache file exists
        if os.path.exists(data_mat_path):
            with open(data_mat_path, 'rb') as f:
                data_mat = pickle.load(f)
            print("Load existed {} matrix cache successfully!".format(set_tag))
            return data_mat
        # load data matrix from json file
        print("Load data matrix from {}...".format(data_folder))
        data_mat = []
        if not os.path.exists(data_folder):
            print("[Dataset] {} folder not exists!".format(data_folder))
            return None
        json_files = os.listdir(data_folder)
        if len(json_files) < 1:
            print("[Dataset] No json file found in {}!".format(data_folder))
            return None
        for idx in range(len(json_files)):
            print("Load json file {} ({}/{})".format(json_files[idx], idx + 1, len(json_files)))
            data_raw = self.load_json(data_folder + '/' + json_files[idx])
            for data in tqdm(data_raw):
                data_x = data['x']
                data_y = data['y']
                data_x_n = [track['track_id'] for track in data_x]
                # single y for prediction in training set
                if type(data_y) is dict:
                    data_y_n = [data_y['track_id']]
                else:
                    data_y_n = [track['track_id'] for track in data_y]
                data_mat.append({'x': data_x_n, 'y': data_y_n})
        print("{} data loaded: {}".format(set_tag, len(data_mat)))
        # save to cache file
        with open(data_mat_path, 'wb') as f:
            pickle.dump(data_mat, f)
        return data_mat

    # get song matrix from: song dictionary, genre matrix, meta matrix, 
    #                       audio matrix and lyric matrix
    # dim_clip = 'pca'
    # dim_list = [genre_dim, meta_dim, audio_dim, lyric_dim]meta_audio_ 
    def get_song_mat(self):
        song_mat_path = self.outdir + 'song_mat.npy'
        # check if there is song matrix cache exists
        if os.path.exists(song_mat_path):
            song_mat = np.load(song_mat_path)
            print("Load existed song matrix cache successfully!")
            return song_mat
        # load song matrix
        print("No song dictionary cache exists, generate new one...")
        # add an empty column at the beginning
        song_mat = np.zeros((len(self.song_dict), 1))
        if self.genre is True:
            # pca
            if self.dim_list[0] > 0:
                U, S, V = torch.pca_lowrank(torch.tensor(self.genre_mat), q=self.dim_list[0])
                self.genre_mat = torch.matmul(torch.tensor(self.genre_mat), V[:, :self.dim_list[0]]).numpy()
            song_mat = np.hstack([song_mat, self.genre_mat])
        if self.meta is True:
            # pca
            if self.dim_list[1] > 0:
                U, S, V = torch.pca_lowrank(torch.tensor(self.meta_mat), q=self.dim_list[1])
                self.meta_mat = torch.matmul(torch.tensor(self.meta_mat), V[:, :self.dim_list[1]]).numpy()
            song_mat = np.hstack([song_mat, self.meta_mat])
        if self.audio is not None:
            # pca
            if self.dim_list[2] > 0:
                U, S, V = torch.pca_lowrank(torch.tensor(self.audio_mat), q=self.dim_list[2])
                self.audio_mat = torch.matmul(torch.tensor(self.audio_mat), V[:, :self.dim_list[2]]).numpy()
            song_mat = np.hstack([song_mat, self.audio_mat])
        if self.lyric is not None:
            # pca (torch)
            if self.dim_list[3] > 0:
                U, S, V = torch.pca_lowrank(torch.tensor(self.lyric_mat), q=self.dim_list[3])
                self.lyric_mat = torch.matmul(torch.tensor(self.lyric_mat), V[:, :self.dim_list[3]]).numpy()
            # pca (sklearn)
            # if self.dim_list[3] > 0:
            #     pca = PCA(n_components=self.dim_list[3])
            #     self.lyric_mat = pca.fit_transform(self.lyric_mat)
            song_mat = np.hstack([song_mat, self.lyric_mat])
        # delete the first column
        song_mat = song_mat[:, 1:]
        # save to cache file
        np.save(song_mat_path, song_mat)
        return song_mat
    
    # get train and test song matrix: set_tag = 'train' or 'test'
    def get_train_test_song_mat(self, song_json_path, set_tag='train'):
        set_song_dict_path = self.outdir + '{}_song_dict.npy'.format(set_tag)
        set_song_mat_path = self.outdir + '{}_song_mat.npy'.format(set_tag)
        # check if train cache file exists
        if os.path.exists(set_song_dict_path) and os.path.exists(set_song_mat_path):
            set_song_dict = np.load(set_song_dict_path, allow_pickle='TRUE').item()
            set_song_mat = np.load(set_song_mat_path)
            print("Load existed {} song dictionary and matrix cache successfully!".format(set_tag))
            return set_song_dict, set_song_mat
        # load song dict
        print("No {} song dictionary cache exists, load it from {}...".format(set_tag, song_json_path))
        set_song_dict = {}
        set_song_json = self.load_json(song_json_path)
        print("Load {} song dictionary...".format(set_tag))
        for track_id in tqdm(set_song_json.keys()):
            set_song_dict[track_id] = len(set_song_dict)
        print("Load {} song matrix...".format(set_tag))
        set_song_mat = np.zeros((len(set_song_dict), self.song_mat.shape[1]))
        for track_id in tqdm(set_song_dict.keys()):
            set_song_mat[set_song_dict[track_id], :] = self.song_mat[self.song_dict[track_id], :]
        # save to cache file
        np.save(set_song_dict_path, set_song_dict)
        np.save(set_song_mat_path, set_song_mat)
        return set_song_dict, set_song_mat
        
    # load song json as a dictionary: key: track id, value: track index for other matrices
    def load_song_dict(self):
        song_dict_path = self.outdir + song_dict_name
        song_old_new_dict_path = self.outdir + song_old_new_dict_name
        # check if there is song dictionary cache exists
        if os.path.exists(song_dict_path) and os.path.exists(song_old_new_dict_path):
            song_dict = np.load(song_dict_path, allow_pickle='TRUE').item()
            song_old_new_dict = np.load(song_old_new_dict_path, allow_pickle='TRUE').item()
            print("Load existed song dictionary cache successfully!")
            return song_dict, song_old_new_dict
        # load song dict from song json file
        print("No song dictionary cache exists, load it from {}...".format(self.song_json_url))
        song_dict = {}
        song_old_new_dict = {}
        for track_id in tqdm(self.song_json.keys()):
            song_dict[track_id] = len(song_dict)
            song_old_new_dict[track_id] = self.song_json[track_id]['is_new']
        # save song dict as cache file
        np.save(song_dict_path, song_dict)
        np.save(song_old_new_dict_path, song_old_new_dict)
        return song_dict, song_old_new_dict

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
        genre_mat = np.zeros((len(self.song_dict), len(genre_dict) + 1), dtype=np.float32)
        for track_id in tqdm(self.song_dict.keys()):
            tags = json.loads(self.song_json[track_id][tag])
            if tags is None:
                continue
            for genre_tag in tags.keys():
                genre_mat[self.song_dict[track_id], genre_dict[genre_tag]] = float(tags[genre_tag]) / 100
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
                meta_mat[self.song_dict[track_id], meta_dict[meta_key]] = meta_info[meta_key]
        # normalize the features
        #meta_mat = meta_mat - meta_mat.min(axis=0)
        #meta_mat[meta_mat < 1e-9] = 1e-9
        # normalize
        # meta_mat = (meta_mat - meta_mat.min(axis=0)) / meta_mat.ptp(axis=0)
        # standardize
        meta_mat = (meta_mat - meta_mat.mean(axis=0)) / meta_mat.std(axis=0)

        # expand the dimension of meta features from 13 to 130
        meta_mat = np.tile(meta_mat, (1, 10))

        # save meta matrix as cache file
        np.save(meta_mat_path, meta_mat)
        return meta_mat

    # load audio features extracted from pre-trained model, e.g: musicnn
    # merge (method to merge sequence): 'avg': average pooling, 'max': max pooling
    def load_audio_features(self, merge='max'):
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
                    audio_mat[self.song_dict[ori_track_id], :] = track_mat
                else:
                    unfound_track_counter += 1
                    print("[Audio feature]Unfound track No.{}: {}".format(unfound_track_counter, ori_track_id))
                    continue
        
        # old version
        # sum_of_dim = audio_mat.sum(axis=0)
        # audio_mat = audio_mat / sum_of_dim[np.newaxis, :]
        
        # normalize the dim 0
        # ptp = audio_mat.ptp(axis=0)
        # mat_min = audio_mat.min(axis=0)
        # ptp[ptp < 1e-9] = 1e-9
        # audio_mat = audio_mat - mat_min / ptp[np.newaxis, :]

        # normalize the whole block
        # ptp = audio_mat.ptp()
        # mat_min = audio_mat.min()
        # ptp = 1e-9 if ptp < 1e-9 else ptp
        # audio_mat = audio_mat - mat_min / ptp

        # normalize among dim 1 (final choice)
        ptp = audio_mat.ptp(axis=1)
        ptp[ptp < 1e-9] = 1e-9
        ptp = np.reshape(ptp, (audio_mat.shape[0], 1))
        ptp = np.repeat(ptp, audio_mat.shape[1], axis=1)
        min_mat = np.reshape(audio_mat.min(axis=1), (audio_mat.shape[0], 1))
        min_mat = np.repeat(min_mat, audio_mat.shape[1], axis=1)
        audio_mat = (audio_mat - min_mat) / ptp
        
        # standardize among dim 0
        # std = audio_mat.std(axis=0)
        # std[std < 1e-9] = 1e-9
        # audio_mat = (audio_mat - audio_mat.mean(axis=0)) / std
        # standardize among dim 1
        # std = audio_mat.std(axis=1)
        # std[std < 1e-9] = 1e-9
        # std = np.reshape(std, (audio_mat.shape[0], 1))
        # std = np.repeat(std, audio_mat.shape[1], axis=1)
        # mean_mat = np.reshape(audio_mat.min(axis=1), (audio_mat.shape[0], 1))
        # mean_mat = np.repeat(mean_mat, audio_mat.shape[1], axis=1)
        # audio_mat = (audio_mat - mean_mat) / std
        # save audio matrix to cache file
        np.save(audio_mat_path, audio_mat)
        return audio_mat

    # load lyric features
    def load_lyric_features(self):
        lyric_mat_path = self.outdir + '{}_mat.npy'.format(self.lyric)
        # check if there is cached matrix
        if os.path.exists(lyric_mat_path):
            lyric_mat = np.load(lyric_mat_path)
            print("Load existed lyric feature matrix cache successfully!")
            return lyric_mat
        # load lyric feature matrix from lyric feature folder
        print("No lyric feature cache file exists, load it from extracted features...")
        lyric_feature_root = self.dataset_root + '/' + out_folder + \
            '/{}_features'.format(self.lyric)
        # load one matrix to get the shape
        if not os.path.exists(lyric_feature_root):
            print("Fail to find extracted lyric features: {} not exists".format(lyric_feature_root))
            return None
        lyric_feature_files = os.listdir(lyric_feature_root)
        if len(lyric_feature_files) < 1:
            print("No extracted lyric feature file exists in {}".format(lyric_feature_root))
            return None
        lyric_feature_sample = np.load(lyric_feature_root + '/' + lyric_feature_files[0])
        lyric_feature_shape = lyric_feature_sample.shape
        # load lyric feature names
        if self.lyric == 'tf_idf':
            self.lyric_feature_names = np.load(lyric_feature_root + '/feature_names.npy', allow_pickle=True)
            lyric_mat = np.zeros((len(self.song_dict), lyric_feature_shape[1]))
            unfound_track_counter = 0
            for track_id in tqdm(self.song_dict.keys()):
                ori_track_id = track_id
                # convert track_id to file name
                if not track_id.find("spotify") == -1:
                    track_id = track_id.split(":")[2]
                track_id += '.npy'
                # make sure the file exists
                if track_id in lyric_feature_files:
                    track_mat = np.load(lyric_feature_root + '/' + track_id)
                    # save to audio matrix
                    lyric_mat[self.song_dict[ori_track_id], :] = track_mat
                else:
                    unfound_track_counter += 1
                    print("[Lyric feature]Unfound track No.{}: {}".format(unfound_track_counter, ori_track_id))
                    continue
            # normalize among dim 0
            # ptp = lyric_mat.ptp(axis=0)
            # min_val = lyric_mat.min(axis=0)
            # # replace zeros with epsilon
            # ptp[ptp < 1e-9] = 1e-9
            # lyric_mat = lyric_mat - min_val / ptp[np.newaxis, :]

            # normalize among dim 1
            ptp = lyric_mat.ptp(axis=1)
            ptp[ptp < 1e-9] = 1e-9
            ptp = np.reshape(ptp, (lyric_mat.shape[0], 1))
            ptp = np.repeat(ptp, lyric_mat.shape[1], axis=1)
            min_mat = np.reshape(lyric_mat.min(axis=1), (lyric_mat.shape[0], 1))
            min_mat = np.repeat(min_mat, lyric_mat.shape[1], axis=1)
            lyric_mat = (lyric_mat - min_mat) / ptp

            # save audio matrix to cache file
            np.save(lyric_mat_path, lyric_mat)
            return lyric_mat
        elif self.lyric == 'glove':
            lyric_mat = np.zeros((len(self.song_dict), lyric_feature_shape[1]))
            unfound_track_counter = 0
            for track_id in tqdm(self.song_dict.keys()):
                ori_track_id = track_id
                # convert track_id to file name
                if not track_id.find("spotify") == -1:
                    track_id = track_id.split(":")[2]
                track_weights_pkl = track_id + '.pkl'
                track_id += '.npy'
                # make sure the file exists
                if track_id in lyric_feature_files:
                    track_mat = np.load(lyric_feature_root + '/' + track_id)
                    # weighted average mixture
                    # load weights, which is the frequency of the word token
                    track_weights_f = open(lyric_feature_root + '/' + track_weights_pkl, 'rb')
                    track_weights_raw = pickle.load(track_weights_f)
                    track_weights = np.array([weight[1] for weight in track_weights_raw])
                    track_weights_sum = track_weights.sum()
                    track_weights = np.reshape(track_weights / track_weights_sum, (track_weights.shape[0], 1))
                    track_weights = np.repeat(track_weights, track_mat.shape[1], axis=1)
                    track_mat = track_mat * track_weights
                    track_mat = track_mat.sum(axis=0)
                    # mean mixture
                    # track_mat = track_mat.mean(axis=0)
                    # max mixture
                    # track_mat = track_mat.max(axis=0)
                    # save to audio matrix
                    lyric_mat[self.song_dict[ori_track_id], :] = track_mat
                else:
                    unfound_track_counter += 1
                    print("[Lyric feature]Unfound track No.{}: {}".format(unfound_track_counter, ori_track_id))
                    continue
            # # normalize among dim 0
            # sum_of_dim = lyric_mat.sum(axis=0)
            # # replace zeros with epsilon
            # sum_of_dim[sum_of_dim < 1e-9] = 1e-9
            # lyric_mat = lyric_mat / sum_of_dim[np.newaxis, :]
            # save audio matrix to cache file
            np.save(lyric_mat_path, lyric_mat)
            return lyric_mat



if __name__ == '__main__':
    dataset = Dataset(dataset_root='E:/dataset_old')
    print(dataset.get_dim())
    [x_train_len, y_train_len, x_train_mat, y_train_mat] = dataset.get_data(set_tag='train')
    x_len_max, y_len_max = dataset.get_max_seq_len()
    x_len_list, y_len_list, x_mat_tensor_list, y_mat_tensor_list = dataset.get_batched_data([x_train_len, y_train_len, x_train_mat, y_train_mat])
    print(x_len_max, y_len_max)
    print("Finished!")