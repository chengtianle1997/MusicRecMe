# train.py
# To train our music recommendation model

from data_loader import data_loader

import os
import logging
import logger # my logger class
import datetime
import argparse

import model
import torch
import torch.nn as nn

# Params
batch_size = 50
learning_rate = 1e-3

echo_nest_sub_path = 'dataset/echo_nest/sub_data'
echo_nest_whole_path = 'dataset/echo_nest/data'

# get task name from args configurations
def get_task_name(genre=False, meta=True, audio='musicnn', lyric=None):
    task_name = ""
    if genre is True:
        task_name += "genre_"
    if meta is True:
        task_name += "meta_"
    if audio is None:
        task_name += "none" + '_'
    else:
        task_name += audio + '_'
    if lyric is None:
        task_name += "none"
    else:
        task_name += lyric
    return task_name

# get time strformat string from datetime.time
def get_time_string(time):
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def train(args):
    if args.sub:
        work_folder_root = args.root + '/' + echo_nest_sub_path
    else:
        work_folder_root = args.root + '/' + echo_nest_whole_path
    # create work folder
    task_name = get_task_name(args.gen, args.meta, args.audio, args.lyric)
    work_folder = work_folder_root + '/' + task_name
    if not os.path.exists(work_folder):
        os.makedirs(work_folder)
    # init logger
    time_stamp = datetime.datetime.now()
    time_string = get_time_string(time_stamp)
    log = logger.logger(work_folder, time=time_stamp)
    # load training data
    dataset = data_loader.Dataset(dataset_root='E:', sub=args.sub, genre=args.gen, meta=args.meta, \
        audio=args.audio, lyric=args.lyric)
    music_embed_dim, music_embed_dim_list = dataset.get_dim()
    log.print("dataset loaded:")
    log.print("music embed dim: {} [{}, {}, {}, {}]".format(music_embed_dim, music_embed_dim_list[0], \
        music_embed_dim_list[1], music_embed_dim_list[2], music_embed_dim_list[3]))
    # load train and valid set
    train_data_list = dataset.get_data(set_tag='train')
    x_train_len_list, y_train_len_list, x_train_tensor_list, y_train_tensor_list = \
        dataset.get_batched_data(train_data_list)
    valid_data_list = dataset.get_data(set_tag='valid')
    x_valid_len_list, y_valid_len_list, x_valid_tensor_list, y_valid_tensor_list = \
        dataset.get_batched_data(valid_data_list)
    train_len, valid_len = len(x_train_len_list) * batch_size, len(x_valid_len_list) * batch_size
    log.print("{} playlists found (train: {}, valid: {})".format(train_len + valid_len, train_len, valid_len))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train music recommendation model')

    # dataset path
    parser.add_argument('--echo', '-e', action='store_true', help='Load Echo Nest Dataset')
    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')
    parser.add_argument('--sub', '-s', action='store_true', help='Generate sub-dataset for debugging and testing')
    # features
    parser.add_argument('--gen', '-g', action='store_true', help='Including Genre features')
    parser.add_argument('--meta', '-m', action='store_true', help='Including Meta features')
    parser.add_argument('--audio', '-a', type=str, default='musicnn', help='Name of audio features')
    parser.add_argument('--lyric', '-l', type=str, default='None', help='Name of lyric features')
    # training params
    parser.add_argument('--batch', '-b', type=int, default=50, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')

    args = parser.parse_args()

    # check None values
    if args.audio == "None":
        args.audio = None
    if args.lyric == 'None':
        args.lyric = None

    train(args)