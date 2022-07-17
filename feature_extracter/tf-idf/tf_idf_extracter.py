# tf_idf_extracter.py
# Extract document-level TF-IDF features from lyric text

import os
from tqdm import tqdm
import numpy as np
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer

# raw music file folder
in_folder = 'dataset/base/lyric_raw'
out_folder = 'dataset/base/tf_idf_features_128'

def extract_feature(in_folder, out_folder):
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
    # vectorizer
    vectorizer = TfidfVectorizer(input='filename', max_features=128)
    # fit
    # for test:
    # in_files = in_files[:10000]
    print("Vectorizer fitting...")
    in_files_full_path = [in_folder + '/' + file for file in in_files]
    vectorizer.fit(in_files_full_path)
    feature_names = vectorizer.get_feature_names_out()
    # save feature names
    np.save(out_folder + '/' + 'feature_names.npy', feature_names)
    # transform
    print("Transforming...")
    for in_file in tqdm(in_files):
        out_vector = vectorizer.transform([in_folder + '/' + in_file]).toarray().astype(np.float32)
        out_file_path = out_folder + '/' + in_file.split('.')[0] + '.npy'
        np.save(out_file_path, out_vector)
    print("Lyric TF-IDF feature extraction complete!")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Tf-idf feature from lyric texts')

    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')

    args = parser.parse_args()

    in_folder = args.root + '/' + in_folder
    out_folder = args.root + '/' + out_folder

    extract_feature(in_folder, out_folder)