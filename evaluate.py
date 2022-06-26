# evaluate.py
# evaluate the music recommender

from data_loader import data_loader

import os
import logging
import logger # my logger class
import datetime
import argparse
import pickle

import model as Model
import torch
import torch.nn as nn
import torch.optim as optim

import visdom
import numpy as np



def evaluate(args):
    # check if it is a sub-dataset for debugging and testing
    if args.sub:
        work_folder_root = args.root + '/' + echo_nest_sub_path
    else:
        work_folder_root = args.root + '/' + echo_nest_whole_path
    # get work folder path
    task_name = get_task_name(args.gen, args.meta, args.audio, args.lyric)
    work_folder = work_folder_root + '/' + task_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate music recommendation model on both valid and test set')

    # dataset path
    parser.add_argument('--echo', '-e', action='store_true', help='Load Echo Nest Dataset')
    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')
    parser.add_argument('--sub', '-s', action='store_true', help='Generate sub-dataset for debugging and testing')
    # features
    parser.add_argument('--gen', '-g', action='store_true', help='Including Genre features')
    parser.add_argument('--meta', '-m', action='store_true', help='Including Meta features')
    parser.add_argument('--audio', '-a', type=str, default='musicnn', help='Name of audio features')
    parser.add_argument('--lyric', '-l', type=str, default='None', help='Name of lyric features')
    
    args = parser.parse_args()

    # check None values
    if args.audio == "None":
        args.audio = None
    if args.lyric == 'None':
        args.lyric = None

    evaluate(args)