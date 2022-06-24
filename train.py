# train.py
# To train our music recommendation model

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

# Params
batch_size = 50
learning_rate = 1e-3
early_stop_steps = 50 # early stopping triggered if over early_stop_steps no update
loss_type = 'seq_cos' # 'cos' or 'rmse' or 'seq_cos'
loss_type_list = ['rmse', 'cos', 'seq_cos']
seq_k = 10
use_music_embedding = False
include_x_loss = True

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

# move optimizer to cuda (gpu)
def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def update_visualizer(vis, opts):
    batch_size, train_loss_batch, train_loss_epoch, valid_loss_epoch, recall_epoch, top_track_loss \
        = opts[0], opts[1], opts[2], opts[3], opts[4], opts[5]
    # draw the train loss plot among batches
    vis.line(
        X=np.array([i for i in range(len(train_loss_batch))]) / batch_size,
        Y=train_loss_batch,
        win='train_loss_batch',
        name='Train',
        update=None,
        opts={
            'showlegend': True,
            'title': "Training loss among batches",
            'x_label': 'Epoch',
            'y_label': 'Loss'
        }
    )
    # draw the train and valid loss plot among epochs
    vis.line(
        X=[i for i in range(len(train_loss_epoch))],
        Y=train_loss_epoch,
        win='train_valid_loss',
        name='Train',
        update=None,
        opts={
            'showlegend': True,
            'title': "Training and Validation loss",
            'x_label': 'Epoch',
            'y_label': 'Loss'
        }
    )
    vis.line(
        X=[i for i in range(len(train_loss_epoch))],
        Y=valid_loss_epoch,
        win='train_valid_loss',
        name='Valid',
        update='append',
    )
    vis.line(
        X=[i for i in range(len(top_track_loss))],
        Y=top_track_loss,
        win='train_valid_loss',
        name='Recom',
        update='append',
    )
    # draw recall lines
    recall_epoch_np = np.array(recall_epoch)
    recall_10 = recall_epoch_np[:, 0]
    recall_50 = recall_epoch_np[:, 1]
    recall_100 = recall_epoch_np[:, 2]
    vis.line(
        X=[i for i in range(len(recall_epoch))],
        Y=recall_10,
        win='recall_plot',
        name='Recall@10',
        update=None,
        opts={
            'showlegend': True,
            'title': "Recall rate @ 10, 50, and 100",
            'x_label': 'Epoch',
            'y_label': 'Recall'
        }
    )
    vis.line(
        X=[i for i in range(len(recall_epoch))],
        Y=recall_50,
        win='recall_plot',
        name='Recall@50',
        update='append'
    )
    vis.line(
        X=[i for i in range(len(recall_epoch))],
        Y=recall_100,
        win='recall_plot',
        name='Recall@100',
        update='append'
    )

def train(args):
    # initialize visualizer
    vis = visdom.Visdom()
    # load batch size and learning rate
    batch_size = args.batch
    learning_rate = args.lr
    # check if it is a sub-dataset for debugging and testing
    if args.sub:
        work_folder_root = args.root + '/' + echo_nest_sub_path
    else:
        work_folder_root = args.root + '/' + echo_nest_whole_path
    # create work folder
    task_name = get_task_name(args.gen, args.meta, args.audio, args.lyric)
    work_folder = work_folder_root + '/' + task_name
    if not os.path.exists(work_folder):
        os.makedirs(work_folder)
    # create cache folder
    cache_folder = work_folder + '/cache'
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    
    # init logger
    time_stamp = datetime.datetime.now()
    time_string = get_time_string(time_stamp)
    log = logger.logger(work_folder, time=time_stamp)
    
    # load training data
    dataset = data_loader.Dataset(dataset_root='E:', sub=args.sub, genre=args.gen, meta=args.meta, \
        audio=args.audio, lyric=args.lyric, outdir=cache_folder)
    music_embed_dim, music_embed_dim_list = dataset.get_dim()
    log.print("dataset loaded:")
    log.print("music embed dim: {} [{}, {}, {}, {}]".format(music_embed_dim, music_embed_dim_list[0], \
        music_embed_dim_list[1], music_embed_dim_list[2], music_embed_dim_list[3]))
    # load train and valid set
    train_data_list = dataset.get_data(set_tag='train')
    x_train_len_list, y_train_len_list, x_train_tensor_list, y_train_tensor_list = \
        dataset.get_batched_data(train_data_list, batch_size=batch_size, fix_length=False)
    # there is no need to batch valid set, we avoid batching by setting batch_size = the size of valid set
    valid_data_list = dataset.get_data(set_tag='valid')
    valid_data_batch =  len(valid_data_list[0])
    x_valid_len_list, y_valid_len_list, x_valid_tensor_list, y_valid_tensor_list = \
        dataset.get_batched_data(valid_data_list, batch_size=valid_data_batch, fix_length=False)
    train_len, valid_len = len(x_train_len_list) * batch_size, len(x_valid_len_list) * valid_data_batch
    log.print("{} playlists found (train: {}, valid: {})".format(train_len + valid_len, train_len, valid_len))
    # get track id list for valid set
    x_valid_tracks = valid_data_list[2]
    y_valid_tracks = valid_data_list[3]
    
    # check loss type
    if loss_type not in loss_type_list:
        log.print("Cannot support loss function type: {}".format(loss_type))
        exit
    # load model
    model = Model.UserAttention(music_embed_dim, music_embed_dim_list, \
        return_seq=True if loss_type=='seq_cos' else False, seq_k=seq_k, re_embed=use_music_embedding)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss(reduction='none') # do not calculate mean or sum
    cos_sim_loss = nn.CosineSimilarity(dim=2) # cosine similarity on embedding dimension
    seq_cos_loss = Model.SequenceEmbedLoss()
    start_epoch = 0
    best_epoch = 0
    best_valid_loss = float('inf')
    best_recall_50 = 0
    # check if checkpoint exists
    sub_task_name = get_sub_task_name(args)
    checkpoint_final_path = work_folder + '/' + sub_task_name + '.pt'
    if os.path.isfile(checkpoint_final_path):
        checkpoint = torch.load(checkpoint_final_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_valid_loss = checkpoint['valid_loss']
        log.print("Checkpoint found, start from epoch {}, loss: {}, valid loss:{}"\
            .format(start_epoch, loss, best_valid_loss))
    
    # move model to device
    model = nn.DataParallel(model) # designed for multi-GPUs
    model = model.to(device)
    optimizer_to(optimizer, device)

    # load recommender
    if loss_type == 'seq_cos':
        recommender = Model.MusicRecommenderSequenceEmbed(dataset, device, mode='train', \
            model=model, use_music_embedding=use_music_embedding)
    else:
        recommender = Model.MusicRecommender(dataset, device, mode='train', model=model, \
            use_music_embedding=use_music_embedding)

    # loss recorder
    train_loss_batch = []
    train_loss_epoch = []
    valid_loss_epoch = []
    recall_epoch = []
    top_track_loss_epoch = []
    train_process_recorder_path = work_folder + '/' + sub_task_name + '.pkl'
    if os.path.exists(train_process_recorder_path):
        with open(train_process_recorder_path, 'rb') as f:
            recorders = pickle.load(f)
            train_loss_batch = recorders['train_loss_batch']
            train_loss_epoch = recorders['train_loss_epoch']
            valid_loss_epoch = recorders['valid_loss_epoch']
            recall_epoch = recorders['recall_epoch']
            top_track_loss_epoch = recorders['top_track_loss']
    
    # start training --------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epoch):
        # iterate through batch
        for i, data in enumerate(x_train_tensor_list):
            # get training data
            x = data.to(device)  # [batch_size, max_seq_len, music_embed_dim]
            x_len = x_train_len_list[i]  # [batch_size]
            y = y_train_tensor_list[i].to(device)  # [batch_size, max_seq_len, music_embed_dim]
            y_len = y_train_len_list[i]
            # zero the parameter gradients
            optimizer.zero_grad()
            # generate mask for attention
            x_mask = Model.generate_mask(x_len).to(device)
            # generate mask for y to calculate loss
            y_mask = Model.generate_out_mask(y_len).to(device)
            x_y_mask = Model.generate_out_mask(x_len).to(device)
            # forward, pred is user embedding
            model.train()
            pred = model(x, x_mask)
            # calculate loss
            if loss_type == 'rmse':
                loss = Model.get_rmse_loss(mse_loss, pred, y, y_mask, model=model if use_music_embedding else None, \
                    x=x if include_x_loss else None, x_mask=x_y_mask if include_x_loss else None)
            elif loss_type == 'cos':
                loss = Model.get_cosine_sim_loss(cos_sim_loss, pred, y, y_mask, model=model if use_music_embedding else None, \
                    x=x if include_x_loss else None, x_mask=x_y_mask if include_x_loss else None)
            elif loss_type == 'seq_cos':
                loss = seq_cos_loss(pred, y, y_mask, model=model if use_music_embedding else None, \
                    x=x if include_x_loss else None, x_mask=x_y_mask if include_x_loss else None)
            # back propagation
            loss.backward()
            optimizer.step()
            train_loss_batch.append(loss.item())
            # log.print("Epoch: {}, Batch: {}, train loss: {}".format(epoch, i, loss.item()))
        
        # valid 
        x_valid = x_valid_tensor_list[0].to(device)
        x_valid_len = x_valid_len_list[0]
        y_valid = y_valid_tensor_list[0].to(device)
        y_valid_len = y_valid_len_list[0]
        # generate mask for attention
        x_valid_mask = Model.generate_mask(x_valid_len).to(device)
        # generate mask for y to calculate loss
        y_valid_mask = Model.generate_out_mask(y_valid_len).to(device)
        x_y_valid_mask = Model.generate_out_mask(x_valid_len).to(device)
        # prediction
        model.eval()
        pred_valid = model(x_valid, x_valid_mask) 
        # calculate valid loss
        if loss_type == 'rmse':
            valid_loss = Model.get_rmse_loss(mse_loss, pred_valid, y_valid, y_valid_mask, model=model if use_music_embedding else None, \
                x=x_valid if include_x_loss else None, x_mask=x_y_valid_mask if include_x_loss else None)
        elif loss_type == 'cos':
            valid_loss = Model.get_cosine_sim_loss(cos_sim_loss, pred_valid, y_valid, y_valid_mask, model=model if use_music_embedding else None, \
                x=x_valid if include_x_loss else None, x_mask=x_y_valid_mask if include_x_loss else None)
        elif loss_type == 'seq_cos':
            valid_loss = seq_cos_loss(pred_valid, y_valid, y_valid_mask, model=model if use_music_embedding else None, \
                    x=x_valid if include_x_loss else None, x_mask=x_y_valid_mask if include_x_loss else None)
        train_loss_epoch.append(train_loss_batch[-1])
        valid_loss_epoch.append(valid_loss.item())
        
        # recommendation
        top_10_track_ids, top_10_track_mats, recalls = recommender.recommend(pred_valid, x_valid_tracks, y_valid_tracks, return_songs=True)
        recall_epoch.append(recalls)
        # calculate loss for recommended tracks
        if loss_type == 'rmse':
            top_track_loss = Model.get_rmse_loss(mse_loss, top_10_track_mats[:, 0, :], \
                y_valid, y_valid_mask, model=model if use_music_embedding else None)
        elif loss_type == 'cos':
            top_track_loss = Model.get_cosine_sim_loss(cos_sim_loss, top_10_track_mats[:, 0, :], \
                y_valid, y_valid_mask, model=model if use_music_embedding else None)
        elif loss_type == 'seq_cos':
            top_track_loss = seq_cos_loss(top_10_track_mats, y_valid, y_valid_mask, \
                model=model if use_music_embedding else None)
        top_track_loss_epoch.append(top_track_loss.item())
        log.print("[Epoch: {}] train loss: {}, valid loss: {}, recalls: (@10: {}, @50: {}, @100: {})"\
            .format(epoch, train_loss_batch[-1], valid_loss, recalls[0], recalls[1], recalls[2]))
        
        # update visualizer
        opts = [batch_size, train_loss_batch, train_loss_epoch, valid_loss_epoch, recall_epoch, top_track_loss_epoch]
        update_visualizer(vis, opts)
        
        # check if it is a better model
        if valid_loss < best_valid_loss or recalls[1] > best_recall_50:
            # save the model
            torch.save({
                'epoch': epoch,
                'loss': train_loss_batch[-1],
                'valid_loss': valid_loss,
                'model_state_dict': model.module.state_dict(), # for data parallel model
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_final_path)
            # update best_epoch and loss
            best_valid_loss = valid_loss
            best_recall_50 = recalls[1]
            best_epoch = epoch
        
        log.print("Best Epoch: {}, valid loss: {}, best recall@50: {}".format(best_epoch, best_valid_loss, best_recall_50))

        # early stopping
        if epoch - best_epoch > early_stop_steps:
            log.print("Early Stopping: No valid loss update after {} steps".format(early_stop_steps))
            break
        
        # save the training process
        if epoch % 5 == 0:
            with open(train_process_recorder_path, 'wb') as f:
                pickle.dump({
                    'train_loss_batch': train_loss_batch,
                    'train_loss_epoch': train_loss_epoch,
                    'valid_loss_epoch': valid_loss_epoch,
                    'recall_epoch': recall_epoch,
                    'top_track_loss': top_track_loss_epoch
                }, f)



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
    parser.add_argument('--head', type=int, default=1, help='Number of heads in Multi-head self-attention')
    parser.add_argument('--epoch', type=int, default=500, help='Number of epochs in training process')

    args = parser.parse_args()

    # check None values
    if args.audio == "None":
        args.audio = None
    if args.lyric == 'None':
        args.lyric = None

    train(args)