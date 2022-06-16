# dataout.py
# Generate dataset json or txt file that can be used by our model from mysql database

import musicdb
from tqdm import tqdm
from tqdm.auto import trange
import os
import json
import langTag
import datetime

# Init database
db = musicdb.MusicDB()

# Path to generated dataset
echo_nest_sub_path = 'dataset/echo_nest/sub_data'
echo_nest_whole_path = 'dataset/echo_nest/data'

# Path to raw dataset
_music_path = 'dataset/base/music'
_lyric_path = 'dataset/base/lyric_raw'

MAX_LINE_LIMIT = 100
MIN_PLAYLIST_LEN = 10

# check if the song is a new song
def is_new(time, thres):
    if time is None:
        return False
    time_date = datetime.datetime.strptime(time, '%Y-%m-%d').date()
    thres_date = datetime.datetime.strptime(thres, '%Y-%m-%d').date()
    return time_date >= thres_date

# dictionary format for song json and filtered echo nest table
def get_filtered_song_dict(song_dict, thres, is_valid):
    return {
        "track_id": song_dict["track_id"],
        "track_name": song_dict["track_name"],
        "album_id": song_dict["album_id"],
        "album_name": song_dict["album_name"],
        "artist_id": song_dict["artist_id"],
        "artist_name": song_dict["artist_name"],
        "release_date": song_dict["release_date"],
        "is_new": 1 if is_new(song_dict["release_date"], thres) else 0,
        "audio_features": song_dict["audio_features"],
        "is_valid": is_valid 
    }

# validate echo nest raw table, and update echo nest filtered table
def valid_echo_nest_pre(dataset_root, line_num=1000, old_new_gap='2018-01-01', sub=True):
    # use the previous dataset
    db = musicdb.MusicDB('musicdb')
    # set path to each folder
    music_path = dataset_root + '/' + _music_path
    lyric_path = dataset_root + '/' + _lyric_path
    out_path = dataset_root + '/'
    if sub is True:
        out_path += echo_nest_sub_path
    else:
        out_path += echo_nest_whole_path
    # train, valid and test folder
    train_path = out_path + '/train'
    valid_path = out_path + '/valid'
    test_path = out_path + '/test'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    # song table
    song_json_url = out_path + '/song.json'
    # song json: key -> track id, value -> song dict
    song_json = {}
    try:
        jsonf = open(song_json_url)
    except:
        print("Fatal: Cannot open song json file {}".format(song_json_url))
        print("Try to Create one, please restart the program!")
        open(song_json_url, 'w').close()
        return
    
    # load song json
    try:
        song_json = json.load(jsonf)
    except:
        print("No song json data from file: {}".format(song_json_url))
    
    # start generating
    pbar = tqdm(total=line_num)
    line_count = 0
    valid_line_count = 0
    start_line = 0
    while valid_line_count < line_num:
        # Get rows from echo nest table
        rows = db.read_echo_nest(start_line, MAX_LINE_LIMIT)
        start_line += MAX_LINE_LIMIT
        if rows is None:
            print("There is noting to read from echo nest raw table!")
            return
        # iterate rows: each row is a playlist
        for row in rows:
            playlist = json.loads(row['playlist'])
            
            # check if playlist exists
            if db.check_if_user_exists_filter(row['user_id']):
                continue

            new_playlist = []
            audio_lyric = 0
            old_count = 0
            new_count = 0

            for idx in trange(len(playlist), desc='Playlist {}/{}'.format(line_count, line_num)):
                song = playlist[idx]
                # check if it has already been checked and exist in song json
                track_id = song['track_id']
                if track_id in song_json:
                    # track id has been checked, just load and update info from song table
                    song_dict = db.select_song_spotify(track_id)
                    playlist[idx] = song_dict
                    if int(song_dict["audio"]) == 1 and int(song_dict["lyric"]) == 1:
                        audio_lyric += 1
                    # lang_tag = song_dict["lang"]
                    # audio = song_dict["audio"]
                    # lyric = song_dict["lyric"]
                    if song_json[track_id]['is_valid'] == 1:
                        new_playlist.append(song_json[track_id])
                        if song_json[track_id]['is_new'] == 1:
                            new_count += 1
                        else:
                            old_count += 1
                else:
                    # Information check and song table update -------------------
                    # It is here to make sure the playlist info are always up-to-date, 
                    # but it may slow down the process, if you think it it unnecessary, just ignore the code below

                    # check if audio and lyric exists
                    track_file = track_id
                    audio = 0
                    lyric = 0
                    # get row from song table
                    song_dict = db.select_song_spotify(track_id)
                    # check if it is a spotify id or a echo nest id
                    if not track_id.find("spotify") == -1:
                        track_file = track_id.split(":")[2]
                    if os.path.exists(music_path + '/' + track_file + '.mp3'):
                        audio = 1
                    if os.path.exists(lyric_path + '/' + track_file + '.txt'):
                        lyric = 1
                    # update counter audio_lyric
                    if audio == 1 and lyric == 1:
                        audio_lyric += 1
                    # check the language tag
                    lang_tag = None
                    if song_dict["lang"] is not None and song_dict["lang"] is not "null":
                        lang_tag = song_dict["lang"]
                    else:
                        # check lyric language
                        lang_tag = langTag.get_lang_tag(lyric_path + '/' + track_file + '.txt')
                        song_dict["lang"] = lang_tag
                    # check if necessary to update song table
                    if song_dict["lang"] is None or song_dict["lang"] == "null":
                        db.add_lang_to_song(track_id, lang_tag)
                    if audio != int(song_dict["audio"]) or lyric != int(song_dict["lyric"]):
                        db.update_audio_lyric_stat(track_id, audio, lyric)
                        song_dict["audio"] = audio
                        song_dict["lyric"] = lyric
                    # substitude the original playlist: 
                    playlist[idx] = song_dict
                    # ---------------------------------------------------------------
                
                    # Condition filter ----------------------------------------------
                    # check if it is a English song with audio and lyric
                    if lang_tag == 'en' and audio == 1 and lyric == 1:
                        song_json_idx = get_filtered_song_dict(song_dict, old_new_gap, True)
                        new_playlist.append(song_json_idx)
                        if song_json_idx['is_new'] == 1:
                            new_count += 1
                        else:
                            old_count += 1
                    else:
                        # for invalid song
                        song_json_idx = get_filtered_song_dict(song_dict, old_new_gap, False)
                    # save to song json
                    song_json[track_id] = song_json_idx
                    # dump into json file
                    if idx % 20 == 0:
                        try:
                            jsonf = open(song_json_url, 'w+')
                            json.dump(song_json, jsonf)
                            jsonf.close()
                        except:
                            print("Cannot save song json file!")
                        
                    # ---------------------------------------------------------------

            # Check playlist
            try:
                jsonf = open(song_json_url, 'w+')
                json.dump(song_json, jsonf)
                jsonf.close()
            except:
                print("Cannot save song json file!")

            if len(new_playlist) >= MIN_PLAYLIST_LEN:
                db.insert_echo_nest_filter((row['user_id'], json.dumps(new_playlist), len(new_playlist), \
                    old_count, new_count))
                valid_line_count += 1
                pbar.update(1)
            else:
                print("User id {}: playlist too short with {} songs [old: {}, new: {}]"\
                    .format(row['user_id'], len(new_playlist), old_count, new_count))
            # update echo nest raw table
            db.update_echo_nest(row['user_id'], json.dumps(playlist), audio_lyric, \
                float(audio_lyric) / len(playlist), 1)
            # update counter
            line_count += 1  
    pbar.close()  

class data_json(object):
    def __init__(self, path, identifier='train', max_length=100):
        self.path = path
        self.file_counter = 0
        self.max_length = max_length
        self.iden = identifier
        self.content = []
    
    # get exist data volumn
    def exist_num(self):
        file_list = os.listdir(self.path)
        counter = 0
        if len(file_list) == 0:
            return 0
        for file in file_list:
            with open(os.path.join(self.path, file), 'r', encoding='utf-8') as f:
                rdict = json.load(f)
                counter += len(rdict)
        return counter

    # insert sample into json files
    def insert(self, sample):
        # insert sample
        self.content.append(sample)
        # check if content list is full
        if len(self.content) >= self.max_length:
            file_name = "{}/{}_{}-{}.json".format(self.path, self.iden, self.file_counter * self.max_length, \
                self.file_counter * self.max_length + self.max_length - 1)
            with open(file_name, 'w+', encoding='utf-8') as f:
                json.dump(self.content, f)
                self.file_counter += 1
                self.content = []

    # save the rest samples
    def save(self):
        if len(self.content) > 0:
            file_name = "{}/{}_{}-{}.json".format(self.path, self.iden, self.file_counter * self.max_length, \
                self.file_counter * self.max_length + len(self.content) - 1)
            with open(file_name, 'w+', encoding='utf-8') as f:
                json.dump(self.content, f)
                self.file_counter += 1
                self.content = []
    
        

# sort the songs in playlist according to release date
# return:
# playlist: list of songs in uses' playlist
# q: the gap of old song and new song, position p is the last old song
def arrange_playlist(playlist, old_new_gap):
    # make sure new songs all in the end, using double pointers
    p = 0
    q = len(playlist) - 1
    while p < q:
        if is_new(playlist[p]['release_date'], old_new_gap):
        # or, we can use the is_new column directly
        # if playlist[p]["is_new"] == 1:
            # exchange with a old song at the tail
            while is_new(playlist[q]['release_date'], old_new_gap):
                q -= 1
            if p >= q:
                break
            # exchange p and q
            playlist[p], playlist[q] = playlist[q], playlist[p]
        p += 1
    # ensure position p is an old song
    if is_new(playlist[p]['release_date'], old_new_gap):
        p -= 1
    return playlist, p

# save song json file: save song json train and song json test
def save_song_json(url, song_json):
    try:
        jsonf = open(url, 'w+')
        json.dump(song_json, jsonf)
        jsonf.close()
    except:
        print("Cannot save song json file!")

MIN_LENGTH_OLD = 5

# generate json file from echo nest filter table
def generate_dataset_pre(dataset_root, line_num=None, old_new_gap='2018-01-01', sub=True):
    # use the previous dataset
    db = musicdb.MusicDB('musicdb')
    # set path to each folder
    music_path = dataset_root + '/' + _music_path
    lyric_path = dataset_root + '/' + _lyric_path
    out_path = dataset_root + '/'
    if sub is True:
        out_path += echo_nest_sub_path
    else:
        out_path += echo_nest_whole_path
    # train, valid and test folder
    train_path = out_path + '/train'
    valid_path = out_path + '/valid'
    test_path = out_path + '/test'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # song table
    song_json_url = out_path + '/song.json'
    song_json_train_url = out_path + '/train_song.json'
    song_json_test_url = out_path + '/test_song.json'
    # song json: key -> track id, value -> song dict, including tracks in train, valid and test set
    song_json = {}
    jsonf = None
    try:
        jsonf = open(song_json_url)
    except:
        print("Fatal: Cannot open song json file {}, try to Create one!".format(song_json_url))
        open(song_json_url, 'w').close()
    # load song json train
    if jsonf is not None:
        try:
            song_json = json.load(jsonf)
        except:

            print("No song json data from file: {}".format(song_json_url))
    # song json train: key -> track id, value -> song dict, including tracks in train and valid set
    song_json_train = {}
    jsonf_train = None
    try:
        jsonf_train = open(song_json_train_url)
    except:
        print("Fatal: Cannot open song json file {}, try to Create one!".format(song_json_train_url))
        open(song_json_train_url, 'w').close()
    # load song json train
    if jsonf_train is not None:
        try:
            song_json_train = json.load(jsonf_train)
        except:

            print("No song json data from file: {}".format(song_json_train_url))
    # song json test: key -> track id, value -> song dict, including tracks in test set
    song_json_test = {}
    jsonf_test = None
    try:
        jsonf_test = open(song_json_test_url)
    except:
        print("Fatal: Cannot open song json file {}, try to create one!".format(song_json_test_url))
        open(song_json_test_url, 'w').close()
    # load song json test
    if jsonf_test is not None:
        try:
            song_json_test = json.load(jsonf_test)
        except:
            print("No song json data from file: {}".format(song_json_test_url))
    
    # init counter
    start_line = 0
    line_counter = 0
    line_valid_counter = 0
    # get the line number
    if line_num is None:
        line_num = db.read_echo_nest_filter_count(MIN_LENGTH_OLD)
    # calculate train, valid, test distribution
    train_num = int(line_num * 0.8)
    valid_num = int(line_num * 0.1)
    test_num = line_num - train_num - valid_num
    # init train, valid, test json
    train_json = data_json(train_path, identifier='train')
    valid_json = data_json(valid_path, identifier='valid')
    test_json = data_json(test_path, identifier='test')
    # start from the last progress
    #line_valid_counter = train_json.exist_num() + valid_json.exist_num() + test_json.exist_num()
    #start_line = line_valid_counter
    # set processing bar
    pbar = tqdm(total=line_num)
    # start generating, line_num is None means read all data from the table
    while line_valid_counter < line_num:
        # read rows from echo nest filter table
        rows = db.read_echo_nest_filter_sorted(start_line, MAX_LINE_LIMIT, MIN_LENGTH_OLD)
        if rows is None:
            break
        start_line += MAX_LINE_LIMIT
        for row in rows:
            # # make sure it has more than 5 old song
            # if not row['old'] >= MIN_LENGTH_OLD:
            #     line_counter += 1 
            #     continue
            # read one playlist, re-arrange it
            playlist = json.loads(row['playlist'])
            playlist, end_pos = arrange_playlist(playlist, old_new_gap)
            # insert the playlist to json file
            # For training data
            if line_valid_counter < train_num:
                res_dict = {}
                res_dict['x'] = playlist[0: int(0.8 * end_pos)]
                res_dict['y'] = playlist[int(0.8 * end_pos): end_pos + 1]
                train_json.insert(res_dict)
                # save songs to song json train
                for song in playlist[0: end_pos + 1]:
                    if not song['track_id'] in song_json_train.keys():
                        song_json_train[song['track_id']] = song
                #save_song_json(song_json_train_url, song_json_train)

            elif line_valid_counter >= train_num and line_valid_counter < train_num + valid_num:
                if line_valid_counter == train_num:
                    # save the last train json
                    train_json.save()
                res_dict = {}
                res_dict['x'] = playlist[0: int(0.8 * end_pos)]
                res_dict['y'] = playlist[int(0.8 * end_pos): end_pos + 1]
                valid_json.insert(res_dict)
                # save songs to song json train
                for song in playlist[0: end_pos + 1]:
                    if not song['track_id'] in song_json_train.keys():
                        song_json_train[song['track_id']] = song
                #save_song_json(song_json_train_url, song_json_train)

            else:
                if line_valid_counter == train_num + valid_num:
                    # save the last valid json
                    valid_json.save()
                res_dict = {}
                res_dict['x'] = playlist[0: int(0.8 * end_pos)]
                res_dict['y'] = playlist[int(0.8 * end_pos):]
                test_json.insert(res_dict)
                # save songs to song json test
                for song in playlist:
                    if not song['track_id'] in song_json_test.keys():
                        song_json_test[song['track_id']] = song
                #save_song_json(song_json_test_url, song_json_test)
            
            line_valid_counter += 1
            pbar.update(1)
    
    # merge train song json and test song json to generate song json for all
    song_json = {**song_json_train, **song_json_test}

    train_json.save()
    valid_json.save()
    test_json.save()

    save_song_json(song_json_url, song_json)
    save_song_json(song_json_train_url, song_json_train)
    save_song_json(song_json_test_url, song_json_test)

    return


if __name__ == '__main__':
    
    # valid_echo_nest('E:')
    generate_dataset_pre('E:/dataset_old', line_num=None)