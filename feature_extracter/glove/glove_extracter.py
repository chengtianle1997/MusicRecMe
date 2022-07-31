# glove_extracter.py
# Extract document-level glove embedding features from lyric text

import os
from tqdm import tqdm
import numpy as np
import argparse
import string
import pickle

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer

# download neccessary nltk tools, run it for the first time
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# raw music file folder
in_folder = 'dataset/base/lyric_raw'
out_folder = 'dataset/base/glove_all_features'

root_folder = 'dataset/base'

use_top_words = False

class GloveEmbedding(object):
    def __init__(self, root_folder, embed='glove.840B.300d'):
        # check if cache exists
        cache_file_name = root_folder + '/' + embed + '.pkl'
        if os.path.exists(cache_file_name):
            # load it from cache
            f = open(cache_file_name, 'rb')
            self.embed_dict = pickle.load(f)
        else:
            # load it from embedding txt file
            embed_file_name = root_folder + '/' + embed + '.txt'
            # open embedding file
            try:
                f = open(embed_file_name, 'r', encoding='utf-8')
            except:
                print("[Fatal] Cannot find Glove embedding file {}".format(embed_file_name))
                exit(0)
            # load embedding dictionary
            self.embed_dict = {}
            print("Load Glove embedding {}".format(embed))
            for line in tqdm(f):
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype=np.float32)
                self.embed_dict[word] = coefs
            print("Glove embedding loaded!")
            # save to cache file
            f_out = open(cache_file_name, 'wb')
            pickle.dump(self.embed_dict, f_out)
            f_out.close()
            f.close()
        # get embedding dimension
        self.embed_dim = self.embed_dict[list(self.embed_dict.keys())[0]].shape[0]
    

    def embed(self, tokens):
        length = len(tokens)
        embed_arr = np.zeros((length, self.embed_dim))
        for i, token in enumerate(tokens):
            token = token[0]
            if token in self.embed_dict.keys():
                embed_arr[i, :] = self.embed_dict[token]
        # embed_arr = embed_arr.flatten()
        return embed_arr
        

def get_most_freq_tokens(tokens, k=50):
    freq = nltk.FreqDist(tokens)
    return freq.most_common(k)
        

def extract_feature(in_folder, out_folder, root_folder, embed='glove.840B.300d'):
    try:
        in_files = os.listdir(in_folder)
        print("{} files found in {}".format(len(in_files), in_folder))
    except:
        print("No file found in {}".format(in_folder))
        return
    # check if the out_folder exists
    out_files = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        out_files = os.listdir(out_folder)
    # pre-pocess file name to keep the track id only
    _in_files = [x.split('.')[0] for x in in_files]
    _out_files = [x.split('.')[0] for x in out_files]
    in_set = set(_in_files)
    out_set = set(_out_files)
    inter_set = in_set & out_set
    in_set = in_set - inter_set
    in_set_list = list(in_set)
    # iterate through lyric raw files
    no_lyric_counter = 0
    # get stop words list in English
    en_stopwords = stopwords.words("english")
    # get stemmer
    stemmer = SnowballStemmer('english')
    # get glove embedding
    glove = GloveEmbedding(root_folder, embed=embed)
    for in_file in tqdm(in_set):
        in_file_name = in_folder + '/' + in_file + '.txt'
        out_file_name = out_folder + '/' + in_file + '.npy'
        out_freq_name = out_folder + '/' + in_file + '.pkl'
        # read txt from lyric raw file
        try:
            f = open(in_file_name, 'r', encoding='utf-8')
            txt = f.read()[:-5]  # remove the last useless word
        except:
            no_lyric_counter += 1
            print("[Warning: {}] Cannot read file {}.".format(no_lyric_counter, in_file_name))
        # remove '\n'
        txt = txt.replace('\n', '. ')
        # convert to lowercase
        txt = txt.lower()
        # sentence tokenizing
        # sent_tokens = sent_tokenize(txt)
        # word tokenizing
        word_tokens = word_tokenize(txt)
        # removing numbers, punctuations, and stopwords
        tokens = [token for token in word_tokens \
            if not token.isdigit() and \
               not token in string.punctuation and \
               not token in en_stopwords]
        # stemming tokens
        # tokens = [stemmer.stem(token) for token in tokens]
        if use_top_words:
            # get most frequent
            most_freq_tokens = get_most_freq_tokens(tokens)
            # get glove embedding
            glove_embed = glove.embed(most_freq_tokens)
            # save most frequent tokens and their frequency
            with open(out_freq_name, 'wb') as f:
                pickle.dump(most_freq_tokens, f)
        else:
            # keep all word emnbeddings
            glove_embed = glove.embed(tokens)
            # take the mean of all word embeddings
            glove_embed = glove_embed.mean(axis=0)
        # save embedding to npy file
        np.save(out_file_name, glove_embed)
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Glove embedding features from lyric texts')

    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')
    parser.add_argument('--embed', '-e', type=str, default='glove.840B.300d', help='Dataset root Path (the path where the dataset folder in)')

    args = parser.parse_args()

    in_folder = args.root + '/' + in_folder
    out_folder = args.root + '/' + out_folder

    root_folder = args.root + '/' + root_folder

    extract_feature(in_folder, out_folder, root_folder, args.embed)