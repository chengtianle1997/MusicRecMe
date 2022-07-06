# musicnn_extracter.py
# musicnn feature extracter: extract audio features from audio file
# adapted from: musicnn/extractor.py

import os
import numpy as np
import librosa

import tensorflow as tf
# disable eager mode for tf.v1 compatibility with tf.v2
tf.compat.v1.disable_eager_execution()

from musicnn import models
from musicnn import configuration as config

from musicnn.extractor import extractor, batch_data
from musicnn.tagger import top_tags

from tqdm import tqdm

import argparse

# raw music file folder
in_folder = 'dataset/base/music'
out_folder = 'dataset/base/musicnn_features'

# file_name = './audio/joram-moments_of_clarity-08-solipsism-59-88.mp3'
# file_name = './audio/TRWJAZW128F42760DD_test.mp3'

# tags = top_tags(file_name, model='MTT_musicnn')

# taggram, tags, features = extractor(file_name, model='MTT_musicnn', extract_features=True)

# print(features.keys())

class extractor(object):
    def __init__(self, model='MSD_musicnn', input_length=3, input_overlap=False):
        # save param
        self.input_length = input_length
        self.input_overlap = input_overlap
        self.model = model
        # select model
        if 'MTT' in model:
            self.labels = config.MTT_LABELS
        elif 'MSD' in model:
            self.labels = config.MSD_LABELS
        num_classes = len(self.labels)
        
        if 'vgg' in model and input_length != 3:
            raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')
        
        # convert seconds to frames
        self.n_frames = librosa.time_to_frames(self.input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
        if not input_overlap:
            self.overlap = self.n_frames
        else:
            self.overlap = librosa.time_to_frames(self.input_overlap, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP)

        # tensorflow: define the model
        print('define the model...')
        tf.compat.v1.reset_default_graph()
        with tf.name_scope('model'):
            self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_frames, config.N_MELS])
            self.is_training = tf.compat.v1.placeholder(tf.bool)
            if 'vgg' in model:
                y, self.pool1, self.pool2, self.pool3, self.pool4, self.pool5 \
                    = models.define_model(self.x, self.is_training, model, num_classes)
            else:
                y, self.timbral, self.temporal, self.cnn1, self.cnn2, self.cnn3, self.mean_pool, self.max_pool, self.penultimate\
                     = models.define_model(self.x, self.is_training, model, num_classes)
            self.normalized_y = tf.nn.sigmoid(y)

        # tensorflow: loading model
        print('load the model...')
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        try:
            saver.restore(self.sess, os.path.dirname(__file__)+'/' +'musicnn' +'/' +model+'/') 
        except:
            if model == 'MSD_musicnn_big':
                raise ValueError('MSD_musicnn_big model is only available if you install from source: python setup.py install')
            elif model == 'MSD_vgg':
                raise ValueError('MSD_vgg model is still training... will be available soon! :)')

    def extract(self, file_name):
        # set indicator
        extract_features = True
        
        # batching data
        print('Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..', end =" ")
        batch, spectrogram = batch_data(file_name, self.n_frames, self.overlap)

        # tensorflow: extract features and tags
        # ..first batch!
        if extract_features:
            if 'vgg' in self.model:
                extract_vector = [self.normalized_y, self.pool1, self.pool2, self.pool3, self.pool4, self.pool5]
            else:
                extract_vector = [self.normalized_y, self.timbral, self.temporal, self.cnn1, self.cnn2, \
                    self.cnn3, self.mean_pool, self.max_pool, self.penultimate]
        else:
            extract_vector = [self.normalized_y]

        tf_out = self.sess.run(extract_vector, 
                        feed_dict={self.x: batch[:config.BATCH_SIZE], 
                        self.is_training: False})

        if extract_features:
            if 'vgg' in self.model:
                predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
                features = dict()
                features['pool1'] = np.squeeze(pool1_)
                features['pool2'] = np.squeeze(pool2_)
                features['pool3'] = np.squeeze(pool3_)
                features['pool4'] = np.squeeze(pool4_)
                features['pool5'] = np.squeeze(pool5_)
            else:
                predicted_tags, timbral_, temporal_, cnn1_, cnn2_, cnn3_, mean_pool_, max_pool_, penultimate_ = tf_out
                features = dict()
                features['timbral'] = np.squeeze(timbral_)
                features['temporal'] = np.squeeze(temporal_)
                features['cnn1'] = np.squeeze(cnn1_)
                features['cnn2'] = np.squeeze(cnn2_)
                features['cnn3'] = np.squeeze(cnn3_)
                features['mean_pool'] = mean_pool_
                features['max_pool'] = max_pool_
                features['penultimate'] = penultimate_
        else:
            predicted_tags = tf_out[0]

        taggram = np.array(predicted_tags)


        # ..rest of the batches!
        for id_pointer in range(config.BATCH_SIZE, batch.shape[0], config.BATCH_SIZE):

            tf_out = self.sess.run(extract_vector, 
                            feed_dict={self.x: batch[id_pointer:id_pointer+config.BATCH_SIZE], 
                            self.is_training: False})

            if extract_features:
                if 'vgg' in self.model:
                    predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
                    features['pool1'] = np.concatenate((features['pool1'], np.squeeze(pool1_)), axis=0)
                    features['pool2'] = np.concatenate((features['pool2'], np.squeeze(pool2_)), axis=0)
                    features['pool3'] = np.concatenate((features['pool3'], np.squeeze(pool3_)), axis=0)
                    features['pool4'] = np.concatenate((features['pool4'], np.squeeze(pool4_)), axis=0)
                    features['pool5'] = np.concatenate((features['pool5'], np.squeeze(pool5_)), axis=0)
                else:
                    predicted_tags, timbral_, temporal_, midend1_, midend2_, midend3_, mean_pool_, max_pool_, penultimate_ = tf_out
                    features['timbral'] = np.concatenate((features['timbral'], np.squeeze(timbral_)), axis=0)
                    features['temporal'] = np.concatenate((features['temporal'], np.squeeze(temporal_)), axis=0)
                    features['cnn1'] = np.concatenate((features['cnn1'], np.squeeze(cnn1_)), axis=0)
                    features['cnn2'] = np.concatenate((features['cnn2'], np.squeeze(cnn2_)), axis=0)
                    features['cnn3'] = np.concatenate((features['cnn3'], np.squeeze(cnn3_)), axis=0)
                    features['mean_pool'] = np.concatenate((features['mean_pool'], mean_pool_), axis=0)
                    features['max_pool'] = np.concatenate((features['max_pool'], max_pool_), axis=0)
                    features['penultimate'] = np.concatenate((features['penultimate'], penultimate_), axis=0)
            else:
                predicted_tags = tf_out[0]

            taggram = np.concatenate((taggram, np.array(predicted_tags)), axis=0)

        if extract_features:
            return taggram, self.labels, features
        else:
            return taggram, self.labels

    # Close the tensorflow session
    def flush(self):
        self.sess.close()
        print('done!')


def extract_feature(in_folder, out_folder):
    # read files from in_folder
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
    in_files = [x.split('.')[0] for x in in_files]
    out_files = [x.split('.')[0] for x in out_files]
    in_set = set(in_files)
    out_set = set(out_files)
    inter_set = in_set & out_set
    in_set = in_set - inter_set
    # initialize a extractor
    m_extractor = extractor()
    # iterate music file in in_folder
    for in_file in tqdm(in_set):
        in_file_name = in_folder + '/' + in_file + '.mp3'
        out_file_name = out_folder + '/' + in_file + '.npy'
        # feature extraction
        taggram, labels, features = m_extractor.extract(in_file_name)
        # save the feature to numpy .npy file (save the 'max_pool' only, which is the last layer)
        np.save(out_file_name, np.array(features['max_pool']))
        # add in_file to out_files list
        out_files.append(in_file)
    print("Musicnn Feature extraction complete!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract music feature from audio with Musicnn')

    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')

    args = parser.parse_args()

    in_folder = args.root + '/' + in_folder
    out_folder = args.root + '/' + out_folder

    extract_feature(in_folder, out_folder)
