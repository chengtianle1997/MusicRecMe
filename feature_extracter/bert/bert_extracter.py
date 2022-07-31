import torch
import numpy as np
from transformers import BertTokenizer, BertModel

import os
from tqdm import tqdm
import numpy as np
import argparse
import string
import pickle

in_folder = 'dataset/base/lyric_raw'
out_folder = 'dataset/base/bert_features'

root_folder = 'dataset/base'

class BertEmbedding():
    def __init__(self, with_sep=True):
        self.with_sep = with_sep
        self.model = BertModel.from_pretrained('bert-base-uncased', \
            output_hidden_states=True).cuda()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def embedding_line(self, text):
        # Tokenize
        # marked_text = "[CLS] " + text + " [SEP]"
        marked_text = text
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])[:,0:512].cuda()
        segments_tensors = torch.tensor([segments_ids])[:,0:512].cuda()
        # Convert tokens to embeddings
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # Remove the first hidden state, which is the input state
            hidden_states = outputs[2][1:]
        # Get embeddings from the bert layer
        token_embeddings = hidden_states[-1]
        # Collapsing tensor into 1-d
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        if not self.with_sep:
            token_embeddings = token_embeddings[1:-1]
        return token_embeddings.cpu()
    
    def embedding_line_sliding(self, text):
        #marked_text = text
        tokenized_text = self.tokenizer.tokenize(text, add_special_tokens=False)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)
        # Convert inputs to PyTorch tensors
        tokens_tensor = list(torch.tensor([indexed_tokens])[0].split(510))
        segments_tensors = list(torch.tensor([segments_ids])[0].split(510))
        # Add special tokens to the tensors except for the last
        for i in range(len(tokens_tensor) - 1):
            tokens_tensor[i] = torch.cat([
                torch.Tensor([101]), tokens_tensor[i].view(-1), torch.Tensor([102])
            ])
            segments_tensors[i] = torch.cat([
                torch.Tensor([1]), segments_tensors[i].view(-1), torch.Tensor([1])
            ])
            
        # Pading the last tensor
        i = len(tokens_tensor) - 1
        pad_len = 510 - tokens_tensor[i].shape[0]
        if pad_len > 0:
            tokens_tensor[i] = torch.cat([
                torch.Tensor([101]), tokens_tensor[i].view(-1), torch.Tensor([102]), torch.Tensor([0] * pad_len)
            ])
            segments_tensors[i] = torch.cat([
                torch.Tensor([1]), segments_tensors[i].view(-1), torch.Tensor([1]), torch.Tensor([0] * pad_len)
            ])
        # Stack the tensor
        tokens_tensor = torch.stack(tokens_tensor).view(i + 1,512).long().cuda()
        segments_tensors = torch.stack(segments_tensors).view(i + 1,512).float().cuda()
        print(tokens_tensor.size())
        # Convert tokens to embeddings
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            # Remove the first hidden state, which is the input state
            hidden_states = outputs[2][1:]
        # Get embeddings from the bert layer: batch_size * 512
        token_embeddings = hidden_states[-1]
        # Collapsing tensor into 1-d
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        if not self.with_sep:
            token_embeddings = token_embeddings[1:-1]
        return token_embeddings.cpu()

    def encode_batch(self, text):
        #tokenized_text = self.tokenizer.tokenize(text, add_special_tokens=False)
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)
        # Convert inputs to PyTorch tensors
        tokens_tensor = list(torch.tensor([indexed_tokens])[0].split(510))
        segments_tensors = list(torch.tensor([segments_ids])[0].split(510))
        # Add special tokens to the tensors except for the last
        for i in range(len(tokens_tensor) - 1):
            tokens_tensor[i] = torch.cat([
                torch.Tensor([101]), tokens_tensor[i].view(-1), torch.Tensor([102])
            ])
            segments_tensors[i] = torch.cat([
                torch.Tensor([1]), segments_tensors[i].view(-1), torch.Tensor([1])
            ])
            
        # Pading the last tensor
        i = len(tokens_tensor) - 1
        pad_len = 510 - tokens_tensor[i].shape[0]
        if pad_len >= 0:
            tokens_tensor[i] = torch.cat([
                torch.Tensor([101]), tokens_tensor[i].view(-1), torch.Tensor([102]), torch.Tensor([0] * pad_len)
            ])
            segments_tensors[i] = torch.cat([
                torch.Tensor([1]), segments_tensors[i].view(-1), torch.Tensor([1]), torch.Tensor([0] * pad_len)
            ])
        # Stack the tensor
        tokens_tensor = torch.stack(tokens_tensor).view(i + 1,512).long()
        segments_tensors = torch.stack(segments_tensors).view(i + 1,512).float()
        return tokens_tensor, segments_tensors
    
    def decode_batch(self, head_index, token_embeddings, segment_tensors):
        token_embeddings = token_embeddings.cpu()
        segment_tensors = segment_tensors.cpu()
        token_embeddings_list = []
        segment_tensors_list = []
        for i in range(1, len(head_index)):
            token_embeddings_b = torch.Tensor()
            segment_tensors_b = torch.Tensor()
            for j in range(head_index[i - 1], head_index[i]):
                token_embeddings_b = torch.cat([token_embeddings_b, token_embeddings[j, 1:511]]) # remove the segments
                segment_tensors_b = torch.cat([segment_tensors_b, segment_tensors[j, 1:511]])
            token_embeddings_list.append(token_embeddings_b)
            segment_tensors_list.append(segment_tensors_b.float().tolist())
        i = len(head_index) - 1
        token_embeddings_b = torch.Tensor()
        segment_tensors_b = torch.Tensor()
        for j in range(head_index[i], token_embeddings.size()[0]):
            token_embeddings_b = torch.cat([token_embeddings_b, token_embeddings[j, 1:511]]) # remove the segments
            segment_tensors_b = torch.cat([segment_tensors_b, segment_tensors[j, 1:511]])
        token_embeddings_list.append(token_embeddings_b)
        segment_tensors_list.append(segment_tensors_b.float().tolist())
        return token_embeddings_list, segment_tensors_list

    def embedding(self, text_list):
        tokens_tensor_list = torch.Tensor()
        segments_tensors_list = torch.Tensor()
        head_index = []
        for text in text_list:
            tokens_tensor, segments_tensors = self.encode_batch(text)
            head_index.append(tokens_tensor_list.size()[0])
            tokens_tensor_list = torch.cat([tokens_tensor_list, tokens_tensor])
            segments_tensors_list = torch.cat([segments_tensors_list, segments_tensors])
        tokens_tensor_b = tokens_tensor_list.long().cuda()
        segments_tensors_b = segments_tensors_list.float().cuda()
        if tokens_tensor_b.shape[0] > 4:
            tokens_tensor_b = tokens_tensor_b[0:4]
            segments_tensors_b = segments_tensors_b[0:4]
        # Convert tokens to embeddings
        with torch.no_grad():
            outputs = self.model(tokens_tensor_b, segments_tensors_b)
            # Remove the first hidden state, which is the input state
            hidden_states = outputs[2][1:]
        # Get embeddings from the bert layer: batch_size * 512
        token_embeddings = hidden_states[-1]
        # Decode the batch
        token_embeddings_list, segment_tensors_list = self.decode_batch(head_index, token_embeddings, segments_tensors_b)
        # padding
        embed = torch.nn.utils.rnn.pad_sequence(token_embeddings_list, batch_first=True)
        mask = self.segment_pad(segment_tensors_list)
        return embed[0, 0:int(sum(mask[0]))]

    # def embedding(self, text_list):
    #     embed = []
    #     mask = []
    #     # Convert the text list to bert embedding
    #     for text in text_list:
    #         bert_embed = self.embedding_line(text)
    #         embed.append(bert_embed)
    #         mask.append(bert_embed.size()[0])
    #     # Padding the embedding sequence and mask
    #     embed = torch.nn.utils.rnn.pad_sequence(embed, batch_first=True)
    #     mask = self.mask_pad(mask)
    #     return embed, mask

    def mask_pad(self, mask):
        max_length = max(mask)
        mask_pad = []
        for i in range(len(mask)):
            mask_pad.append([1.0] * mask[i] + [0.0] * (max_length - mask[i]))
        return mask_pad

    def segment_pad(self, segment):
        max_length = max([len(seg) for seg in segment])
        for i in range(len(segment)):
            pad_len = max_length - len(segment[i])
            if pad_len > 0:
                segment[i] = [1.0] * len(segment[i]) + [0.0] * pad_len
        return segment


def extract_feature(in_folder, out_folder, root_folder):
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
    # get bert embedding
    bert_embed = BertEmbedding()
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
        # convert to bert embedding
        bert_embedding = bert_embed.embedding([txt])
        # take the mean among all words
        bert_embedding = bert_embedding.mean(axis=0)
        # save embedding to npy file
        np.save(out_file_name, bert_embedding)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract Glove embedding features from lyric texts')

    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')

    args = parser.parse_args()

    in_folder = args.root + '/' + in_folder
    out_folder = args.root + '/' + out_folder

    root_folder = args.root + '/' + root_folder

    extract_feature(in_folder, out_folder, root_folder)