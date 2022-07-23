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

from tqdm import tqdm

import visdom
import numpy as np

# Params: you have to set it as training params
batch_size = 50
loss_type = 'seq_cos' # 'cos' or 'rmse' or 'seq_cos'
loss_type_list = ['rmse', 'cos', 'seq_cos']
seq_k = 10
use_music_embedding = False
include_x_loss = False
use_data_pca = True
check_baseline = False

echo_nest_sub_path = 'dataset/echo_nest/sub_data'
echo_nest_whole_path = 'dataset/echo_nest/data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# get sub task name for sub-folder
def get_sub_task_name(args):
    return "b_{}_lr_{}_head_{}".format(args.batch, args.lr, args.head)

# get time strformat string from datetime.time
def get_time_string(time):
    return time.strftime("%Y_%m_%d_%H_%M_%S")

# mode = ['train', 'test'], the 'train' mode can also be used for valid set
def _evaluate(args, model, x_tensor_list, y_tensor_list, x_len_list, y_len_list, \
    x_tracks, y_tracks, recommender, item_cold=False):
    # check mode
    mode = recommender.mode
    # get loss functions
    mse_loss = nn.MSELoss(reduction='none') # do not calculate mean or sum
    cos_sim_loss = nn.CosineSimilarity(dim=2) # cosine similarity on embedding dimension
    seq_cos_loss = Model.SequenceEmbedLoss()
    # result recorders
    loss_list = []
    top_10_track_ids_list = []
    ground_truth_ids_list = []
    recalls_list = []
    others_list = []
    recalls_old_list = []
    recalls_new_list = []
    # iterate batches
    for i in tqdm(range(len(x_tensor_list))):
        # get x and y
        x = x_tensor_list[i].to(device)
        x_len = x_len_list[i]
        y = y_tensor_list[i].to(device)
        y_len = y_len_list[i]
        _batch_size = x.shape[0]
        # generate mask for attention
        x_mask = Model.generate_mask(x_len).to(device)
        # generate mask for y to calculate loss
        y_mask = Model.generate_out_mask(y_len).to(device)
        x_y_mask = Model.generate_out_mask(x_len).to(device)
        # set to evaluation
        model.eval()
        # prediction
        pred = model(x, x_mask)
        # calculate the loss
        if loss_type == 'rmse':
            loss = Model.get_rmse_loss(mse_loss, pred, y, y_mask, model=model if use_music_embedding else None, \
                x=x if include_x_loss else None, x_mask=x_y_mask if include_x_loss else None)
        elif loss_type == 'cos':
            loss = Model.get_cosine_sim_loss(cos_sim_loss, pred, y, y_mask, model=model if use_music_embedding else None, \
                x=x if include_x_loss else None, x_mask=x_y_mask if include_x_loss else None)
        elif loss_type == 'seq_cos':
            loss = seq_cos_loss(pred, y, y_mask, model=model if use_music_embedding else None, \
                    x=x if include_x_loss else None, x_mask=x_y_mask if include_x_loss else None)
        loss_list.append(loss.item())
        ground_truth_ids_list += y_tracks
        # recommendation
        if mode == 'train':
            top_10_track_ids, top_10_track_mats, recalls, others = \
                recommender.recommend(pred, x_tracks[i * _batch_size: (i + 1)*_batch_size], y_tracks[i * _batch_size: (i + 1)*_batch_size], return_songs=True)
        elif mode == 'test':
            top_10_track_ids, top_10_track_mats, recalls, recalls_old, recalls_new, others = \
                recommender.recommend(pred, x_tracks[i * _batch_size: (i + 1)*_batch_size], y_tracks[i * _batch_size: (i + 1)*_batch_size], return_songs=True)
            recalls_new_list.append(recalls_new)
            recalls_old_list.append(recalls_old)
        

        recalls_list.append(recalls)
        others_list.append(others)
        top_10_track_ids_list += top_10_track_ids
    
    # calculate the mean of those loss and recalls
    res_dict = {}
    loss = sum(loss_list) / len(loss_list)
    recalls = np.array(recalls_list).mean(axis=0)
    others = np.array(others_list).mean(axis=0)
    res_dict['loss'] = loss
    res_dict['recalls'] = recalls
    res_dict['others'] = others
    res_dict['gt'] = ground_truth_ids_list
    res_dict['predict'] = top_10_track_ids_list
    if mode == 'test':
        recalls_new = np.array(recalls_new_list).mean(axis=0)
        recalls_old = np.array(recalls_old_list).mean(axis=0)
        res_dict['recalls_new'] = recalls_new
        res_dict['recalls_old'] = recalls_old
    return res_dict

# tag = 'valid' or 'test'
def display_and_save(res_dict, log, result_path, tag='valid', item_cold=False):
    log.print('-------Evaluation results for {} set-------'.format(tag))
    log.print('Loss: {}'.format(res_dict['loss']))
    log.print('Recalls (All): @10 {}, @50 {}, @100 {}'\
        .format(res_dict['recalls'][0], res_dict['recalls'][1], res_dict['recalls'][2]))
    if item_cold:
        log.print('Recalls (Old): @10 {}, @50 {}, @100 {}'\
            .format(res_dict['recalls_old'][0], res_dict['recalls_old'][1], res_dict['recalls_old'][2]))
        log.print('Recalls (New): @10 {}, @50 {}, @100 {}'\
            .format(res_dict['recalls_new'][0], res_dict['recalls_new'][1], res_dict['recalls_new'][2]))
    log.print('R-prec: {}, NDCG: {}, Clicks: {}'\
        .format(res_dict['others'][2], res_dict['others'][1], res_dict['others'][3]))
    
    # save the res_dict
    with open(result_path + '/' + tag + '.pkl', 'wb') as f:
        pickle.dump(res_dict, f)


def evaluate(args):
    # check if it is a sub-dataset for debugging and testing
    if args.sub:
        work_folder_root = args.root + '/' + echo_nest_sub_path
    else:
        work_folder_root = args.root + '/' + echo_nest_whole_path
    # get work folder path
    task_name = get_task_name(args.gen, args.meta, args.audio, args.lyric)
    work_folder = work_folder_root + '/' + task_name
    # get test result output path
    result_path = work_folder + '/' + 'test'
    if not os.path.exists(work_folder):
        print("Cannot find the work folder {}".format(work_folder))
    # create result path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # create cache folder
    cache_folder = work_folder + '/cache'
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    # init logger
    time_stamp = datetime.datetime.now()
    time_string = get_time_string(time_stamp)
    log = logger.logger(result_path, time=time_stamp)
    
    # load dataset
    dataset = data_loader.Dataset(dataset_root='E:', sub=args.sub, genre=args.gen, meta=args.meta, \
        audio=args.audio, lyric=args.lyric, outdir=cache_folder, \
        dim_list=[0, 0, 200, 0] if use_music_embedding or not use_data_pca else [0, 0, 200, 0])
    music_embed_dim, music_embed_dim_list = dataset.get_dim()
    log.print("dataset loaded:")
    log.print("music embed dim: {} [{}, {}, {}, {}]".format(music_embed_dim, music_embed_dim_list[0], \
        music_embed_dim_list[1], music_embed_dim_list[2], music_embed_dim_list[3]))
    
    # load valid set
    valid_data_list = dataset.get_data(set_tag='valid')
    valid_data_batch =  args.batch
    x_valid_len_list, y_valid_len_list, x_valid_tensor_list, y_valid_tensor_list = \
        dataset.get_batched_data(valid_data_list, batch_size=valid_data_batch, fix_length=False)
    # get track id list for valid and test set
    x_valid_tracks = valid_data_list[2]
    y_valid_tracks = valid_data_list[3]
    
    # load test set
    test_data_dict = dataset.get_data(set_tag='test')
    test_data_batch = args.batch
    test_batched_data_dict = {}
    x_test_tracks_dict = {}
    y_test_tracks_dict = {}
    for key in test_data_dict.keys():
        x_test_len_list, y_test_len_list, x_test_tensor_list, y_test_tensor_list = \
            dataset.get_batched_data(test_data_dict[key], batch_size=test_data_batch, fix_length=False)
        test_batched_data_dict[key] = [x_test_len_list, y_test_len_list, x_test_tensor_list, y_test_tensor_list]
        log.print("{} playlists found for {}".format(len(x_test_len_list) * test_data_batch, key))
        x_test_tracks_dict[key] = test_data_dict[key][2]
        y_test_tracks_dict[key] = test_data_dict[key][3]

    valid_len = len(x_valid_len_list) * valid_data_batch
    log.print("{} playlists found for valid".format(valid_len))
    
    
    # check loss type
    if loss_type not in loss_type_list:
        log.print("Cannot support loss function type: {}".format(loss_type))
        exit
    # load model
    model = Model.UserAttention(music_embed_dim, music_embed_dim_list, \
        return_seq=True if loss_type=='seq_cos' else False, seq_k=seq_k, re_embed=use_music_embedding, \
        check_baseline=check_baseline)
    # check if checkpoint exists
    sub_task_name = get_sub_task_name(args)
    checkpoint_final_path = work_folder + '/' + sub_task_name + '.pt'
    if os.path.isfile(checkpoint_final_path):
        checkpoint = torch.load(checkpoint_final_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_valid_loss = checkpoint['valid_loss']
        log.print("Checkpoint found, start from epoch {}, loss: {}, valid loss:{}"\
            .format(start_epoch, loss, best_valid_loss))
    # move model to device
    model = nn.DataParallel(model) # designed for multi-GPUs
    model = model.to(device)
    # load recommender
    if loss_type == 'seq_cos':
        recommender_valid = Model.MusicRecommenderSequenceEmbed(dataset, device, mode='train', \
            model=model, use_music_embedding=use_music_embedding)
    else:
        recommender_valid = Model.MusicRecommender(dataset, device, mode='train', model=model, \
            use_music_embedding=use_music_embedding)
    
    if loss_type == 'seq_cos':
        recommender_test = Model.MusicRecommenderSequenceEmbed(dataset, device, mode='test', \
            model=model, use_music_embedding=use_music_embedding)
    else:
        recommender_test = Model.MusicRecommender(dataset, device, mode='test', model=model, \
            use_music_embedding=use_music_embedding)
    
    # start evaluating
    # for valid data
    valid_res = \
        _evaluate(args, model, x_valid_tensor_list, y_valid_tensor_list, x_valid_len_list, y_valid_len_list, \
            x_valid_tracks, y_valid_tracks, recommender_valid)
    display_and_save(valid_res, log, result_path, tag='valid')
    
    # for test data
    for key in test_batched_data_dict.keys():
        
        item_cold = False
        if key in ['item_cold', 'user_item_cold']:
            item_cold = True
        
        x_test_len_list, y_test_len_list, x_test_tensor_list, y_test_tensor_list = \
            test_batched_data_dict[key][0], test_batched_data_dict[key][1], test_batched_data_dict[key][2], test_batched_data_dict[key][3]
        x_test_tracks, y_test_tracks = x_test_tracks_dict[key], y_test_tracks_dict[key]
        
        test_res = \
            _evaluate(args, model, x_test_tensor_list, y_test_tensor_list, x_test_len_list, y_test_len_list, \
                x_test_tracks, y_test_tracks, recommender_test)
        
        display_and_save(test_res, log, result_path, tag=key, item_cold=item_cold)
    

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
    # training params
    parser.add_argument('--batch', '-b', type=int, default=50, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--head', type=int, default=1, help='Number of heads in Multi-head self-attention')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs in training process')
    args = parser.parse_args()

    # check None values
    if args.audio == "None":
        args.audio = None
    if args.lyric == 'None':
        args.lyric = None

    evaluate(args)