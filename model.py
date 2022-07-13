# model.py
# Music recommendation model

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1] # head_dim
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k) # [batch_size, num_heads, seq_length, seq_length]
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1) # [batch_size, num_heads, seq_length, seq_length]
    values = torch.matmul(attention, v)        # [batch_size, num_heads, seq_length, head_dim]
    return values, attention

def generate_mask(len_list):
    max_len = max(len_list)
    mask_np = np.zeros((len(len_list), max_len, max_len)) # [batch_size, seq_length, seq_length]
    # mark valid positions
    for i in range(len(len_list)):
        mask_np[i, 0: len_list[i], 0: len_list[i]] = 1
    mask = torch.tensor(mask_np)
    return mask

def generate_out_mask(len_list):
    max_len = max(len_list)
    mask_np = np.zeros((len(len_list), max_len, 1))
    # mark valid positions
    for i in range(len(len_list)):
        mask_np[i, 0: len_list[i], 0] = 1
    mask = torch.tensor(mask_np)
    return mask

# input:
# y_mask = [batch_size, seq_length, 1]
def get_rmse_loss(mse_loss, pred, y, y_mask, model=None, x=None, x_mask=None):
    # include x
    if x is not None and x_mask is not None:
        y = torch.cat([y, x], dim=1)
        y_mask = torch.cat([y_mask, x_mask], dim=1)
    if model is not None:
        # music embedding
        y = model.module.music_embed(y)
    pred = pred.reshape(pred.shape[0], 1, pred.shape[1]) # [batch_size, 1, embed_dim]
    pred = pred.repeat(1, y.shape[1], 1) # [batch_size, max_seq_len, embed_dim]
    y_mask = y_mask.repeat(1, 1, pred.shape[2])
    loss = mse_loss(pred, y)
    loss = (loss * y_mask).sum()
    valid_elements = y_mask.sum()
    loss = torch.sqrt(loss / valid_elements)
    return loss

# input:
# y_mask = [batch_size, seq_length, 1]
def get_cosine_sim_loss(cos_sim_loss, pred, y, y_mask, model=None, x=None, x_mask=None):
    # include x
    if x is not None and x_mask is not None:
        y = torch.cat([y, x], dim=1)
        y_mask = torch.cat([y_mask, x_mask], dim=1)
    if model is not None:
        # music embedding
        y = model.module.music_embed(y)
    pred = pred.reshape(pred.shape[0], 1, pred.shape[1]) # [batch_size, 1, embed_dim]
    pred = pred.repeat(1, y.shape[1], 1) # [batch_size, max_seq_len, embed_dim]
    y_mask = y_mask.reshape(y_mask.shape[0], y_mask.shape[1]) # [batch_size, max_seq_len]
    loss = 1. - cos_sim_loss(pred, y) # [batch_size, max_seq_len]
    loss = (loss * y_mask).sum()
    valid_elements = y_mask.sum()
    #loss = torch.sqrt(loss / valid_elements)
    loss = loss / valid_elements
    return loss

# input:
# y_mask = [batch_size, seq_length, 1]
def get_mix_cosine_sim_loss(cos_sim_loss, pred, y, y_mask, y_neg, model=None, x=None, x_mask=None):
    # include x
    if x is not None and x_mask is not None:
        y = torch.cat([y, x], dim=1)
        y_mask = torch.cat([y_mask, x_mask], dim=1)
    if model is not None:
        # music embedding
        y = model.module.music_embed(y)
        y_neg = model.module.music_embed(y_neg)
    pred = pred.reshape(pred.shape[0], 1, pred.shape[1]) # [batch_size, 1, embed_dim]
    pred = pred.repeat(1, y.shape[1], 1) # [batch_size, max_seq_len, embed_dim]
    y_mask = y_mask.reshape(y_mask.shape[0], y_mask.shape[1]) # [batch_size, max_seq_len]
    loss = 0.2 - cos_sim_loss(pred, y) + cos_sim_loss(pred, y_neg) # [batch_size, max_seq_len]
    loss = loss.clamp(min=0)
    loss = (loss * y_mask).sum()
    valid_elements = y_mask.sum()
    #loss = torch.sqrt(loss / valid_elements)
    #loss = loss / valid_elements
    return loss

class MusicRecommender(object):
    '''
    Recommend songs according to song dictionary and user embedding
    '''
    def __init__(self, dataset, device, model=None, mode='train', use_music_embedding=True):
        self.device = device
        if mode == 'train':
            self.song_dict = dataset.train_song_dict
            self.song_mat_ori = torch.tensor(dataset.train_song_mat).to(device)
        elif mode == 'test':
            self.song_dict = dataset.test_song_dict
            self.song_mat_ori = torch.tensor(dataset.test_song_mat).to(device)
        # song_mat = [song_num, song_embed_dim]
        self.cos_sim = nn.CosineSimilarity(dim=1)
        # reverse song dict to find track by index
        self.rev_song_dict = {v: k for k, v in self.song_dict.items()}
        # get music embedding
        if use_music_embedding is True and model is not None:
            self.model = model
            #self.song_mat = self.model.module.music_embed(self.song_mat)

    
    # recommend songs according to cosine similarity (for multi-users at once)
    # user_embed[]: user embedding from UserAttention model [batch_size, user_embed_dim]
    # x_valid_tracks []: given track ids in x (not embeddings) [batch_size, seq_len]
    # y_valid_tracks []: ground truth track ids (not embeddings) [batch_size, seq_len]
    # return_songs: whether return recommended track ids or not
    # top_k: return top K recommended songs
    def recommend(self, user_embed, x_valid_tracks, y_valid_tracks, model=None, return_songs=False, top_k=100):
        top_10_recall_num = 0
        top_50_recall_num = 0
        top_100_recall_num = 0
        ground_truth_num = 0
        top_10_track_list = []
        ap_list = []
        ndcg_list = []
        # music embedding
        if model is not None:
            self.song_mat = model.module.music_embed(self.song_mat_ori)
        else:
            self.song_mat = self.song_mat_ori

        # top_10_track_mat = [batch_size, 10, embed_dim]
        top_10_track_mat = torch.zeros((user_embed.shape[0], 10, self.song_mat.shape[1])).to(self.device)
        # iterate through users
        for i in range(user_embed.shape[0]):
            # similarity = [song_num]
            similarity = self.cos_sim(self.song_mat, user_embed[i, :].reshape(1, user_embed.shape[1]))
            # mask out songs in x
            index_mask = torch.zeros(similarity.shape[0]).to(self.device)
            for track in x_valid_tracks[i]:
                index_mask[self.song_dict[track]] = 2 # make sure songs in x will not be selected
            similarity = similarity - index_mask
            # top_k_index = [top_k]
            top_k_index = torch.topk(similarity.flatten(), top_k).indices
            # get track embedding
            top_k_embed = [self.song_mat[int(n)] for n in top_k_index]
            # get track ids
            top_k_track = [self.rev_song_dict[int(n)] for n in top_k_index]
            top_10_track_list.append(top_k_track)
            for n in range(10):
                top_10_track_mat[i, n, :] = self.song_mat[int(top_k_index[n])]
            # get intersection of predicted track and groud truth tracks
            gt_set = set(y_valid_tracks[i])
            # top 10, 50, 100 recall
            top_10_inter = set(top_k_track[0:10]) & gt_set
            top_50_inter = set(top_k_track[0:50]) & gt_set
            top_100_inter = set(top_k_track[0:100]) & gt_set
            top_10_recall_num += len(top_10_inter)
            top_50_recall_num += len(top_50_inter)
            top_100_recall_num += len(top_100_inter)
            ground_truth_num += len(gt_set)
            # get ap
            ap_list.append(get_ap(y_valid_tracks[i], top_k_track))
            # get ndcg
            ndcg_list.append(get_ndcg(y_valid_tracks[i], top_k_track))
        
        recalls = [top_10_recall_num / ground_truth_num, top_50_recall_num / ground_truth_num, \
            top_100_recall_num / ground_truth_num]
        
        mean_avg_prec = sum(ap_list) / len(ap_list)
        mean_ndcg = sum(ndcg_list) / len(ndcg_list)

        if return_songs is True:
            return top_10_track_list, top_10_track_mat, recalls, [mean_avg_prec, mean_ndcg]
        else:
            # calculate recall rate
            return recalls, mean_avg_prec

class SequenceEmbedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=3)

    # pred = [batch_size, seq_x_length, embed_dim]
    # y = [batch_size, seq_y_length, embed_dim]
    # y_mask = [batch_size, seq_y_length, 1]
    # x = [batch_size, seq_length, embed_dim]
    # x_mask = [batch_size, seq_length, 1]
    # reduction = 'mean', 'max'
    def forward(self, pred, y, y_mask, model=None, x=None, x_mask=None, reduction='max', neg=False):
        # include x
        if x is not None and x_mask is not None:
            y = torch.cat([y, x], dim=1)
            y_mask = torch.cat([y_mask, x_mask], dim=1)
        # music embedding
        if model is not None:
            y = model.module.music_embed(y)
        batch_size, seq_x_len, embed_dim = pred.shape[0], pred.shape[1], pred.shape[2]
        seq_y_len = y.shape[1]
        pred = pred.reshape(batch_size, seq_x_len, 1, embed_dim)
        y = y.reshape(batch_size, 1, seq_y_len, embed_dim)
        similarity = self.cos_sim(pred, y) # [batch_size, x_len, y_len]
        if reduction == 'max':
            similarity = torch.max(similarity, dim=1)[0] # [batch_size, y_len]
        elif reduction == 'mean':
            similarity = similarity.mean(dim=1)
        loss = 1. - similarity
        # loss = similarity
        y_mask = y_mask.reshape(batch_size, seq_y_len)
        valid_num = y_mask.sum()
        loss = (loss * y_mask).sum() / valid_num
        #return torch.sqrt(loss)
        #return torch.log(loss)
        return loss

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, dataset, device, model=None, mode='train', multi_label=False):
        super().__init__()
        self.multi_label = multi_label
        if multi_label:
            self.loss = torch.nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.device= device
        if mode == 'train':
            self.song_mat_ori = torch.tensor(dataset.train_song_mat).to(device)
        else:
            self.song_mat_ori = torch.tensor(dataset.test_song_mat).to(device)
        self.model = model
        self.song_dim = self.song_mat_ori.shape[0]
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred, x_inv, y, model=None):
        need_max_seq = True
        # fit for non-sequence user embeddings
        if len(pred.shape) < 3:
            pred = pred.reshape(pred.shape[0], 1, pred.shape[1])
            need_max_seq = False
        
        batch_size, seq_x_len, embed_dim = \
             pred.shape[0], pred.shape[1], pred.shape[2]

        if model is not None:
            self.song_mat = model.module.music_embed(self.song_mat_ori)
        else:
            self.song_mat = self.song_mat_ori

        # iterate through user embeddings
        similarity = torch.zeros(batch_size, self.song_mat.shape[0], seq_x_len).to(self.device)
        # iterate through users
        for i in range(batch_size):
            for j in range(seq_x_len):
                # similarity = [batch_size, song_num, x_len]
                similarity[i, :, j] = self.cos_sim(self.song_mat, pred[i, j, :].reshape(1, pred.shape[2]))
        # similarity = [batch_size, song_num, 1]
        if need_max_seq:
            similarity = torch.max(similarity, dim=2)[0]
        else:
            similarity = similarity.reshape(batch_size, -1)
        similarity = similarity * x_inv
        #imilarity = similarity.type(torch.DoubleTensor).to(self.device)
        # calculate cross entropy loss
        # similarity = (similarity + 1.) / 2.
        if self.multi_label:
            # for multi label: more than one song to predict in y
            # using sigmoid instead of softmax
            similarity = torch.sigmoid(similarity * 5)
            loss = self.loss(similarity, y)
            return loss
        else:
            # for single label: only one song to predict in y
            similarity = self.softmax(similarity)
            y = self.softmax(y)
            loss = self.loss(similarity, y)
            return loss


def get_recalls(gt, top_k_track):
    gt_set = set(gt)
    # top 10, 50, 100 recall
    top_10_inter = set(top_k_track[0:10]) & gt_set
    top_50_inter = set(top_k_track[0:50]) & gt_set
    top_100_inter = set(top_k_track[0:100]) & gt_set
    # top_10_recall_num += len(top_10_inter)
    # top_50_recall_num += len(top_50_inter)
    # top_100_recall_num += len(top_100_inter)
    # ground_truth_num += len(gt_set)
    recalls_num = np.array([len(top_10_inter), len(top_50_inter), len(top_100_inter), len(gt_set)])
    return recalls_num

# calculate the average precision
def get_ap(gt, top_k_track, k=10):
    top_k_relav = [1 if track in gt else 0 for track in top_k_track]
    prec_sum = 0
    k_ap = k
    gt_set = set(gt)
    for i in range(min(len(top_k_track), k_ap)):
        # check the precision at this position is valid
        if top_k_relav[i] == 1:
            # calculate the precision
            pred_set = set(top_k_track[0:i + 1])
            inter_set = pred_set & gt_set
            prec_sum += len(inter_set) / (i + 1)
    avg_prec = prec_sum / min(len(top_k_track), k_ap)
    return avg_prec

# calculate the NDCG
def get_ndcg(gt, top_k_track, k=10):
    if k is not None:
        k_ndcg = k
    else:
        k_ndcg = len(top_k_track)
    numi = 0
    domi = 0
    for i in range(k_ndcg):
        inv_rank = 1 + 1 / math.log2(i + 1 + 1)
        domi += inv_rank
        if top_k_track[i] in gt:
            numi += 1 + inv_rank
    return numi / domi

# calculate the R-precision
def get_r_prec(gt, top_k_track):
    num_gt = len(gt)
    gt_set = set(gt)
    rec_set = set(top_k_track[0:num_gt])
    inter_set = gt_set & rec_set
    return len(inter_set) / num_gt

# calculate the clicks
def get_clicks(gt, top_k_track):
    for i in range(0, len(top_k_track)):
        if top_k_track[i] in gt:
            return (i - 1) / 10
    return (len(top_k_track) - 1) / 10


class MusicRecommenderSequenceEmbed(object):
    '''
    Recommend songs according to song dictionary and user embedding
    '''
    def __init__(self, dataset, device, mode='train', model=None, use_music_embedding=False):
        self.device = device
        self.mode = mode
        if mode == 'train':
            self.song_dict = dataset.train_song_dict
            self.song_mat = torch.tensor(dataset.train_song_mat).to(device)
        elif mode == 'test':
            self.song_dict = dataset.test_song_dict
            self.song_mat = torch.tensor(dataset.test_song_mat).to(device)
            self.song_old_new_dict = dataset.song_old_new_dict
        # reshape: song_mat = [song_num, song_embed_dim, 1]
        # self.song_mat = self.song_mat.reshape(self.song_mat.shape[0], self.song_mat.shape[1], 1)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        # reverse song dict to find track by index
        self.rev_song_dict = {v: k for k, v in self.song_dict.items()}
        # get music embedding
        self.use_music_embedding = use_music_embedding
        self.song_mat_ori = self.song_mat.clone()
        if use_music_embedding is True and model is not None:
             self.model = model
        #     self.song_mat = self.model.module.music_embed(self.song_mat_ori)
    
    # recommend songs according to cosine similarity (for multi-users at once)
    # user_embed[]: user embedding from UserAttention model [batch_size, seq_x_len, user_embed_dim]
    # x_valid_tracks []: given track ids in x (not embeddings) [batch_size, seq_x_len]
    # y_valid_tracks []: ground truth track ids (not embeddings) [batch_size, seq_y_len]
    # return_songs: whether return recommended track ids or not
    # top_k: return top K recommended songs
    def recommend(self, user_embed, x_valid_tracks, y_valid_tracks, return_songs=False, model=None, top_k=100):
        # get music embedding
        # if self.use_music_embedding is True and self.model is not None:
        #     self.song_mat = self.model.module.music_embed(self.song_mat_ori)
        # get dimension
        batch_size, seq_x_len, embed_dim = \
            user_embed.shape[0], user_embed.shape[1], user_embed.shape[2]
        # init counters
        top_10_recall_num = 0
        top_50_recall_num = 0
        top_100_recall_num = 0
        ground_truth_num = 0
        # numpy matrix to save recall numbers
        recalls_num_counter = np.zeros(4)
        recalls_old_num_counter = np.zeros(4)
        recalls_new_num_counter = np.zeros(4)
        top_10_track_list = []
        ap_list = []
        ndcg_list = []
        r_prec_list = []
        clicks_list = []
        # music embedding
        if model is not None:
            self.song_mat = model.module.music_embed(self.song_mat_ori)
        else:
            self.song_mat = self.song_mat_ori
        # top_10_track_mat = [batch_size, 10, embed_dim]
        top_10_track_mat = torch.zeros((batch_size, 10, self.song_mat.shape[1])).to(self.device)
        # iterate through users
        for i in range(batch_size):
            # iterate through user embeddings
            similarity = torch.zeros(self.song_mat.shape[0], seq_x_len).to(self.device)
            for j in range(seq_x_len):
                # similarity = [song_num, x_len]
                similarity[:, j] = self.cos_sim(self.song_mat, user_embed[i, j, :].reshape(1, user_embed.shape[2]))
            # similarity = [song_num, 1]
            similarity = torch.max(similarity, dim=1)[0]
            # mask out songs in x
            index_mask = torch.zeros(similarity.shape[0]).to(self.device) 
            for track in x_valid_tracks[i]:
                index_mask[self.song_dict[track]] = 2  # make sure songs in x will not be selected
            similarity = similarity - index_mask
            # top_k_index = [top_k]
            top_k_index = torch.topk(similarity.flatten(), top_k).indices
            # get track embedding
            top_k_embed = [self.song_mat[int(i)] for i in top_k_index]
            # get track ids
            top_k_track = [self.rev_song_dict[int(i)] for i in top_k_index]
            top_10_track_list.append(top_k_track)
            top_10_track_list.append(top_k_track)
            for n in range(10):
                top_10_track_mat[i, n, :] = self.song_mat[int(top_k_index[n])]
            # get intersection of predicted track and groud truth tracks
            recalls_num = get_recalls(y_valid_tracks[i], top_k_track)
            ap_list.append(get_ap(y_valid_tracks[i], top_k_track))
            ndcg_list.append(get_ndcg(y_valid_tracks[i], top_k_track))
            r_prec_list.append(get_r_prec(y_valid_tracks[i], top_k_track))
            clicks_list.append(get_clicks(y_valid_tracks[i], top_k_track))
            # gt_set = set(y_valid_tracks[i])
            # # top 10, 50, 100 recall
            # top_10_inter = set(top_k_track[0:10]) & gt_set
            # top_50_inter = set(top_k_track[0:50]) & gt_set
            # top_100_inter = set(top_k_track[0:100]) & gt_set
            recalls_num_counter = recalls_num_counter + recalls_num
            if self.mode == 'test':
                # recalls for old and new songs
                y_valid_tracks_new = [track for track in y_valid_tracks[i] if self.song_old_new_dict[track]==1]
                y_valid_tracks_old = [track for track in y_valid_tracks[i] if self.song_old_new_dict[track]==0]
                recalls_new = get_recalls(y_valid_tracks_new, top_k_track)
                recalls_old = get_recalls(y_valid_tracks_old, top_k_track)
                recalls_new_num_counter += recalls_new
                recalls_old_num_counter += recalls_old

        recalls = [recalls_num_counter[0] / recalls_num_counter[3], \
                    recalls_num_counter[1] / recalls_num_counter[3], \
                    recalls_num_counter[2] / recalls_num_counter[3]]

        mean_avg_prec = sum(ap_list) / len(ap_list)
        mean_ndcg = sum(ndcg_list) / len(ndcg_list)
        mean_r_prec = sum(r_prec_list) / len(r_prec_list)
        mean_clicks = sum(clicks_list) / len(clicks_list)

        if return_songs is True:
            if self.mode == 'test':
                recalls_new = [recalls_new_num_counter[0] / recalls_new_num_counter[3], \
                    recalls_new_num_counter[1] / recalls_new_num_counter[3], \
                    recalls_new_num_counter[2] / recalls_new_num_counter[3]]
                recalls_old = [recalls_old_num_counter[0] / recalls_old_num_counter[3], \
                    recalls_old_num_counter[1] / recalls_old_num_counter[3], \
                    recalls_old_num_counter[2] / recalls_old_num_counter[3]]
                top_10_track_list, top_10_track_mat
                return top_10_track_list, top_10_track_mat, recalls, recalls_old, recalls_new, mean_avg_prec
            else:
                return top_10_track_list, top_10_track_mat, recalls, [mean_avg_prec, mean_ndcg, mean_r_prec, mean_clicks]
        else:
            # return only the recall rate
            return recalls, [mean_avg_prec, mean_ndcg, mean_r_prec, mean_clicks]

class MultiheadAttention(nn.Module):
    # Multihead Attention
    def __init__(self, input_dim, embed_dim, num_heads):
        '''
        input_dim: input music embedding dimension
        embed_dim: embedding dimension for Q, K and V
        num_heads: number of multi-heads
        '''
        super().__init__()
        assert embed_dim % num_heads == 0, "[Multi-head Attention]: Embedding dimension must be 0 modulko number of heads!"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stock all weights matrices together for efficiency
        # self.qkv_proj = nn.Sequential(nn.Linear(input_dim, embed_dim*3), nn.ReLU())
        # self.o_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.qkv_proj = nn.Sequential(nn.Linear(input_dim, embed_dim*3))
        self.o_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim))

        # reset all params
        self._reset_parameters()

    def _reset_parameters(self):
        # Original transformer initialization
        nn.init.xavier_uniform_(self.qkv_proj[0].weight)
        self.qkv_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj[0].weight)
        self.o_proj[0].bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Seperate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, 3*head_dim]
        q, k, v = qkv.chunk(3, dim=-1) # q, k, v = [batch_size, num_heads, seq_length, head_dim]

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3) # [batch_size, seq_length, num_heads, head_dim]
        values = values.reshape(batch_size, seq_length, embed_dim) # [batch_size, seq_length, embed_dim]
        out = self.o_proj(values) # [batch_size, seq_length, embed_dim]
        # out = values

        if return_attention:
            return out, attention
        else:
            return out

# Attention on playlists to generate user embeddings
class UserAttention(nn.Module):
    '''
    The user embedding model with attention on their playlists
    '''
    def __init__(self, music_embed_dim, music_embed_dim_list, embed_dim=None, \
        dropout=0, num_heads=1, re_embed=False, return_seq=False, seq_k=5, \
        check_baseline=False):
        '''
        music_embed_dim: dimension of music embedding
        music_embed_dim_list []: list of [genre, meta, audio, lyric] dimension
        embed_dim: embedding dimension for Q, K, V projection in multi-head attention
        dropout: dropout rate
        '''
        super().__init__()

        self.out_dim = music_embed_dim
        self.music_embed_dim = music_embed_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.return_seq = return_seq
        self.seq_k = seq_k

        self.re_embed = re_embed
        self.check_baseline = check_baseline

        # re-embed audio and lyric for a smaller attention dim
        if re_embed is True:
            self.music_embed = MusicEmbedding(music_embed_dim, music_embed_dim_list)
            self.music_embed_dim = self.music_embed.out_dim

        if self.embed_dim is None:
            self.embed_dim = self.music_embed_dim
        
        # multi-head self-attention layer
        self.self_attn = MultiheadAttention(self.music_embed_dim, self.embed_dim, self.num_heads)

        # multi linear layers
        pass

        # normalize and dropout layer
        self.attn_out_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.embed_dim / 2), self.embed_dim),
            nn.ReLU()
        )

        # de-embed
        if self.re_embed is True:
            self.de_embed = nn.Sequential(nn.ReLU(), nn.Linear(self.music_embed_dim, self.out_dim))
    
    
    def forward(self, x, mask=None):
        # input x: [batch_size, seq_length, music_embed_dim]
        # mask: [batch_size, seq_length, seq_length]

        # for baseline
        if self.check_baseline:
            return x[:, :self.seq_k, :]

        # convert mask
        if mask is not None:
            mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2]) # [batch_size, 1, seq_length, seq_length]
            mask = mask.repeat(1, self.num_heads, 1, 1) # [batch_size, num_heads, seq_length, seq_length]
        
        # music embedding
        if self.re_embed is True:
            x = self.music_embed(x)

        # attention
        attn_out, attn_weights = self.self_attn(x, mask=mask) # [batch_size, seq_length, embed_dim]
        # residual connection
        attn_out = torch.tanh(attn_out)
        attn_out = self.attn_out_norm(attn_out) # [batch_size, seq_length, embed_dim]
        x = x + self.dropout(attn_out)
        # x = self.ffn(x)

        # de-embed
        # if self.re_embed is True:
        #     x = self.de_embed(x)

        # return sequence embedding by PCA
        if self.return_seq is True:
            # return the first K
            return x[:, 0: self.seq_k, :].view(x.shape[0], self.seq_k, -1)
            
            # return the whole sequence
            # return x[:, :, :]  # CUDA out of memory

            # return principle k
            # x_T = x.transpose(1, 2)  # [batch_size, embed_dim, seq_length]
            # U, S, V = torch.pca_lowrank(x_T, q=self.seq_k)  
            # # return the seq_k main directions
            # x = torch.matmul(x_T, V).transpose(1, 2)
            # return x

        # sum among sequence
        mask_ = mask[:, 0, :, 0].to(torch.float32)
        mask_x = mask_.view(x.shape[0], x.shape[1], 1).repeat(1, 1, x.shape[2])
        x = x * mask_x
        x = x.sum(dim=1) # [batch_size, embed_dim]
        mask_ = mask_.sum(dim=-1).view(x.shape[0], 1).repeat(1, x.shape[1]) # [batch_size, 1]
        x = x / mask_

        # linear layers
        x = self.ffn(x)
        
        return x

# Music re-embedding: Cannot support cpu training
class MusicEmbedding(nn.Module):
    '''
    Embedding for original music representation
    '''
    def __init__(self, music_embed_dim, music_embed_dim_list, target_embed_dim_list=[20, 0, 20, 20], out_embed=40):
        super().__init__()
        # take the original embedding dimension
        self.genre_in, self.meta_in, self.audio_in, self.lyric_in = \
            music_embed_dim_list[0], music_embed_dim_list[1], music_embed_dim_list[2], music_embed_dim_list[3]
        self.genre_out, self.meta_out, self.audio_out, self.lyric_out = \
            target_embed_dim_list[0], target_embed_dim_list[1], target_embed_dim_list[2], target_embed_dim_list[3]
        # embedding
        if self.genre_in > 0 and self.genre_out > 0:
            self.genre_embed = nn.Sequential(nn.Linear(self.genre_in, self.genre_out), nn.ReLU())
        if self.meta_in > 0 and self.meta_out > 0:
            self.meta_embed = nn.Sequential(nn.Linear(self.meta_in, self.meta_out, nn.ReLU()))
        if self.audio_in > 0 and self.audio_out > 0:
            self.audio_embed = nn.Sequential(nn.Linear(self.audio_in, self.audio_out, nn.ReLU()))
        if self.lyric_in > 0 and self.lyric_out > 0:
            self.lyric_embed = nn.Sequential(nn.Linear(self.lyric_in, self.lyric_out, nn.ReLU()))
        # out dimension
        self.embed_out_dim = (self.genre_out if self.genre_out > 0 and self.genre_in > 0 else self.genre_in) +\
            (self.meta_out if self.meta_out > 0 and self.meta_in > 0 else self.meta_in) +\
            (self.audio_out if self.audio_out > 0 and self.audio_in > 0 else self.audio_in) +\
            (self.lyric_out if self.lyric_out > 0 and self.lyric_in > 0 else self.lyric_in)
        self.out_dim = out_embed
        self.out_ffn = nn.Sequential(
            nn.Linear(self.embed_out_dim, int(self.embed_out_dim /2)), 
            nn.ReLU(),
            nn.Linear(int(self.embed_out_dim /2), self.out_dim),
            nn.ReLU()
            )
        
    def forward(self, music_embed):

        # music_embed = [batch, music_in]
        music_embed = music_embed.to(torch.float32)
        batch_size = music_embed.shape[0]

        if len(music_embed.shape) == 2:
            genre_part = music_embed[:, 0: self.genre_in]
            meta_part = music_embed[:, self.genre_in: self.genre_in + self.meta_in]
            audio_part = music_embed[:, \
                self.genre_in + self.meta_in: self.genre_in + self.meta_in + self.audio_in]
            lyric_part = music_embed[:, \
                self.genre_in + self.meta_in + self.audio_in: \
                    self.genre_in + self.meta_in + self.audio_in + self.lyric_in]
            out_embed = torch.zeros((batch_size, 1)).to('cuda')

        elif len(music_embed.shape) == 3:
            genre_part = music_embed[:, :,  0: self.genre_in]
            meta_part = music_embed[:, :,  self.genre_in: self.genre_in + self.meta_in]
            audio_part = music_embed[:, :, \
                self.genre_in + self.meta_in: self.genre_in + self.meta_in + self.audio_in]
            lyric_part = music_embed[:, :, \
                self.genre_in + self.meta_in + self.audio_in: \
                    self.genre_in + self.meta_in + self.audio_in + self.lyric_in]
            out_embed = torch.zeros((batch_size, music_embed.shape[1], 1)).to('cuda')
            
        if self.genre_in > 0 and self.genre_out > 0:
            # genre_embed = [batch, genre_out]
            genre_embed = self.genre_embed(genre_part)
            genre_embed = torch.tanh(genre_embed)
            out_embed = torch.cat([out_embed, genre_embed], dim=-1)
        elif self.genre_in > 0:
            genre_embed = genre_part
            out_embed = torch.cat([out_embed, genre_embed], dim=-1)

        if self.meta_in > 0 and self.meta_out > 0:
            meta_embed = self.meta_embed(meta_part)
            meta_embed = torch.tanh(meta_embed)
            out_embed = torch.cat([out_embed, meta_embed], dim=-1)
        elif self.meta_in > 0:
            meta_embed = meta_part
            out_embed = torch.cat([out_embed, meta_embed], dim=-1)
        
        if self.audio_in > 0 and self.audio_out > 0:
            audio_embed = self.audio_embed(audio_part)
            audio_embed = torch.tanh(audio_embed)
            out_embed = torch.cat([out_embed, audio_embed], dim=-1)
        elif self.audio_in > 0:
            audio_embed = audio_part
            out_embed = torch.cat([out_embed, audio_embed], dim=-1)
        
        if self.lyric_in > 0 and self.lyric_out > 0:
            lyric_embed = self.lyric_embed(lyric_part)
            lyric_embed = torch.tanh(lyric_embed)
            out_embed = torch.cat([out_embed, lyric_embed], dim=-1)
        elif self.lyric_in > 0:
            lyric_embed = lyric_part
            out_embed = torch.cat([out_embed, lyric_embed], dim=-1)

        if len(music_embed.shape) == 2:
            out = out_embed[:, 1:]
        elif len(music_embed.shape) == 3:
            out = out_embed[:, :, 1:]
        
        # linear layers
        out = self.out_ffn(out)
        return out

        