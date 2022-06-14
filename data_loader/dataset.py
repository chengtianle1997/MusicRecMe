# dataset.py
# Load the dataset from files, collect meta-info, audio and lyric from web api

import musicdb
from tqdm import tqdm
from tqdm.auto import trange
import spotifyApi
import lyricGenius
import lastFM
import json
import sys
import getopt
import os
import argparse
from langdetect import detect
import langTag

# Init the database
db = musicdb.MusicDB()

# Init the APIs
sp_api = spotifyApi.Spotify()
ly_api = lyricGenius.LyricGenius()
lf_api = lastFM.lastFM()

# path to dataset files
msd_link_txt_n = 'dataset/unique_tracks.txt'
echo_nest_txt_n = 'dataset/train_triplets.txt'
spotify_path_n = 'dataset/spotify/raw'

# Init param
insert_rows_once = 100

# Load unique_track.txt for million song link table
def load_msd_link(msd_link_txt):
    print("Load unique_track.txt for million song link table...")
    with open(msd_link_txt, 'r+', encoding='utf-8') as f:
        data = f.readlines()
        val = []
        for line in tqdm(data):
            spl = line[:-1].split('<SEP>')
            val.append((spl[0], spl[1], str(spl[2]), str(spl[3])))
            # Insert when there is enough rows
            if len(val) >= insert_rows_once:
                db.insert_msd_link(val)
                row_count = 0
                val = []
        # Insert the rest lines
        if len(val) > 0:
            db.insert_msd_link(val)

# Load train_triplets.txt for echo nest interaction
def load_echo_nest(echo_nest_txt, start_line=0):
    print("Load train_triplets.txt for echo nest interaction...")
    with open(echo_nest_txt, 'r+') as f:
        data = f.readlines()
        # Record the previous user-id
        pre_user_id = None
        playlist = []
        none_audio_lyric = 0
        audio_only = 0
        lyric_only = 0
        audio_lyric = 0
        # Accumulate songs to be added into song table
        sql_val_list = []
        # line counter for continued process
        line_counter = start_line
        valid_line_count = 0
        # read data
        for line in tqdm(data[start_line:]):
            spl = line[:-1].split('\t')
            user_id, song_id, play_count = spl[0], spl[1], spl[2]
            # for the first user-id
            if pre_user_id is None:
                pre_user_id = user_id
            # check if the user_id is the same as the previous user
            if not user_id == pre_user_id:
                # check if pre_user_id exist in database
                user_playlist, counter_list = db.select_echo_nest(pre_user_id)
                if user_playlist is None:
                    # it is a new user, we need to save the previous playlist to database
                    db.insert_echo_nest((pre_user_id, json.dumps(playlist), len(playlist), \
                        none_audio_lyric, audio_only, lyric_only, audio_lyric, audio_lyric / len(playlist)))
                else:
                    # the user exist in database, we need to merge the exist playlist with the new one
                    # user_playlist += playlist # concatenate to create new playlist
                    item_none_audio_lyric = counter_list[1]
                    item_audio_only = counter_list[2]
                    item_lyric_only = counter_list[3]
                    item_audio_lyric = counter_list[4]
                    for item in playlist:
                        if not item in user_playlist:
                            user_playlist.append(item)
                            # update counter
                            if item['audio'] == 1 and item['lyric'] == 1:
                                item_audio_lyric += 1
                            elif item['audio'] == 1:
                                item_audio_only += 1
                            elif item['lyric'] == 1:
                                item_lyric_only += 1
                            else:
                                item_none_audio_lyric += 1
                    # insert it into the database
                    db.insert_echo_nest((pre_user_id, json.dumps(user_playlist), len(user_playlist), \
                        item_none_audio_lyric, item_audio_only, item_lyric_only, item_audio_lyric, \
                            item_audio_lyric / len(user_playlist)))
                # reset pre_user_id and playlist
                pre_user_id = user_id
                playlist = []
                valid_line_count = line_counter
                # reset the counters
                none_audio_lyric = 0
                audio_only = 0
                lyric_only = 0
                audio_lyric = 0

            # check if the song id has existed in song table
            ret = db.select_song(song_id)

            # if the song is not exist in song table,
            # we collect meta-info, audio and lyrics from web api
            if ret is None:
                # get the artist name and song title from msd link table
                artist_name, song_title = db.select_msd_link(song_id)
                # use Spotify API to get audio and audio features
                # sp_res_dict = sp_api.search_track_name(song_title, artist_name) 
                sp_res_dict = sp_api.search_track_name_for_id(song_title, artist_name)
                # check if the song exists in spotify
                if sp_res_dict['track_id'] is not None:
                    # check if the track id exist in song table
                    ret_s = db.select_song_spotify(sp_res_dict['track_id'])
                    if ret_s is not None:
                        # track id exist in song table: update echo song id
                        db.update_echo_song_id(sp_res_dict['track_id'], song_id)
                        ret = ret_s
                    else:
                        # it's a new track id: download preview audio and audio features
                        sp_res_dict = sp_api.download_audio(sp_res_dict)
                        # set echo song id
                        sp_res_dict['echo_song_id'] = song_id
                        # get lyric
                        lg_res = ly_api.search_track(sp_res_dict['track_id'], sp_res_dict['track_name'],\
                        sp_res_dict['artist_name'])
                        # add key to song dictionary
                        sp_res_dict['lyric'] = 1 if lg_res is True else 0
                        # add genre tag
                        genre_raw, genre_top = lf_api.get_tag(sp_res_dict['artist_name'], sp_res_dict['track_name'])
                        sp_res_dict['genre_raw'] = genre_raw
                        sp_res_dict['genre_top'] = genre_top
                        # add language tag
                        lang_tag = None
                        if lg_res is True:
                            track_ly = sp_res_dict['track_id']
                            if not track_ly.find("spotify") == -1:
                                track_ly = track_ly.split(":")[2]
                            track_ly_url = "{}/{}.txt".format(ly_api.lyric_path, track_ly)
                            lang_tag = langTag.get_lang_tag(track_ly_url)
                        sp_res_dict['lang'] = lang_tag
                        # get value format and add it to sql_val_list
                        # ret is a dictionary, while sql_val_list is a () with sql values
                        ret = sp_res_dict
                        sql_val_list.append(db.get_song_val_echo(sp_res_dict))
                else:
                    # Cannot find the track in spotify
                    # Create a template result dictionary
                    sp_res_dict = {
                        "track_id": None,
                        "track_name": song_title,
                        "album_id": None,
                        "album_name": None,
                        "artist_id": None,
                        "artist_name": artist_name,
                        "release_date": None,
                        "audio": 0,
                        "preview_url": None,
                        "audio_features": None
                    }
                    sp_res_dict['echo_song_id'] = song_id  
                    # if there is no track id from spotify, we use echo nest song id here
                    sp_res_dict['track_id'] = song_id  
                    # use Lyrics Genius API to get lyrics
                    lg_res = ly_api.search_track(sp_res_dict['track_id'], sp_res_dict['track_name'],\
                        sp_res_dict['artist_name'])
                    # add key to song dictionary
                    sp_res_dict['lyric'] = 1 if lg_res is True else 0
                    # add genre tag
                    genre_raw, genre_top = lf_api.get_tag(sp_res_dict['artist_name'], sp_res_dict['track_name'])
                    sp_res_dict['genre_raw'] = genre_raw
                    sp_res_dict['genre_top'] = genre_top
                    # add language tag
                    lang_tag = None
                    if lg_res is True:
                        track_ly = sp_res_dict['track_id']
                        if not track_ly.find("spotify") == -1:
                            track_ly = track_ly.split(":")[2]
                        track_ly_url = "{}/{}.txt".format(ly_api.lyric_path, track_ly)
                        lang_tag = langTag.get_lang_tag(track_ly_url)
                    sp_res_dict['lang'] = lang_tag
                    # get value format and add it to sql_val_list
                    # ret is a dictionary, while sql_val_list is a () with sql values
                    ret = sp_res_dict
                    sql_val_list.append(db.get_song_val_echo(sp_res_dict))
                
            # now, the song exists in song table, we add it to the playlist
            playlist.append(ret)
            # update counter
            if ret['audio'] == 1 and ret['lyric'] == 1:
                audio_lyric += 1
            elif ret['audio'] == 1:
                audio_only += 1
            elif ret['lyric'] == 1:
                lyric_only += 1
            else:
                none_audio_lyric += 1

            # check if we need to handle the sql_val_list
            if len(sql_val_list) > 0:
                # insert those songs into song table
                db.insert_song(sql_val_list)
                # clear the list
                sql_val_list = []

            line_counter += 1
            print("<IMPORTANT> Line number {} validated!".format(valid_line_count))

        # insert the rest songs into song table
        if len(sql_val_list) > 0:
            db.insert_song(sql_val_list)

# Load Spotify Million Playlist Dataset
def load_spotify_mpd(spotify_path, start_doc_idx=0, start_playlist_idx=0):
    try:
        paths = os.listdir(spotify_path)
    except:
        print("Cannot read from spotify dataset path, check the url!")
        return
    print("{} json slice files found in {}".format(len(paths), spotify_path))
    for doc_idx in trange(start_doc_idx, len(paths), desc='Document'):
        file_url = os.path.join(spotify_path, paths[doc_idx])
        try:
            f = open(file_url)
        except:
            print("Cannot open file {}".format(file_url))
        # load json file as a python dictionary
        data = json.load(f)
        # export playlists
        pl_list = data['playlists']
        # iterate list of the playlist
        for pidx in trange(start_playlist_idx, len(pl_list), desc='Playlist'):
            # get meta info
            meta_info_json = {
                'collaborative': pl_list[pidx]['collaborative'],
                'modified_at': pl_list[pidx]['modified_at'],
                'num_tracks': pl_list[pidx]['num_tracks'], 
                'num_albums': pl_list[pidx]['num_albums'], 
                'num_followers': pl_list[pidx]['num_followers'], 
                'tracks': pl_list[pidx]['tracks'], 
                'num_edits': pl_list[pidx]['num_edits'], 
                'duration_ms': pl_list[pidx]['duration_ms'], 
                'num_artists': pl_list[pidx]['num_artists']
            }
            # Init a new playlist
            pl_new = []
            none_audio_lyric = 0
            audio_only = 0
            lyric_only = 0
            audio_lyric = 0
            # get playlist id and playlist name
            pid_spotify = pl_list[pidx]['pid']
            p_name = pl_list[pidx]['name']
            # Check if the playlist exist in spotify table
            check_name = db.select_spotify(pid_spotify)
            if check_name is not None and check_name == p_name:
                print("Playlist(id: {}, name: {}) exists!".format(pid_spotify, p_name))
                # continue to load next playlist
                continue
            # iterate the track in the playlist
            for tidx in trange(len(pl_list[pidx]['tracks']), desc='Doc:{}/{}, Pid:{}/{}'.format(doc_idx, len(paths), pidx, len(pl_list))):
                # select the track
                track = pl_list[pidx]['tracks'][tidx]
                # check if the track exist in song table
                ret = db.select_song_spotify(track['track_uri'])
                if ret is None:
                    # track not found in song table: search track id with spotify api
                    ret = sp_api.search_track_id(track['track_uri'])
                    # use Lyrics Genius API to get lyrics
                    lg_res = ly_api.search_track(ret['track_id'], ret['track_name'],\
                        ret['artist_name'])
                    # add key to song dictionary
                    ret['lyric'] = 1 if lg_res is True else 0
                    # add the track to song table
                    sql_val_list = []
                    sql_val_list.append(db.get_song_val(ret))
                    db.insert_song(sql_val_list)
                else:
                    # track exist in song table: check spotify status
                    if ret['is_spotify'] == 0:
                        # update spotify status
                        db.update_spotify_stat(track['track_uri'])
                # add the track to playlist
                pl_new.append(ret)
                # update counter
                if ret['audio'] == 1 and ret['lyric'] == 1:
                    audio_lyric += 1
                elif ret['audio'] == 1:
                    audio_only += 1
                elif ret['lyric'] == 1:
                    lyric_only += 1
                else:
                    none_audio_lyric += 1
            # add the playlist to spotify table
            db.insert_spotify((pid_spotify, p_name, json.dumps(pl_new), json.dumps(meta_info_json), len(pl_new), \
                none_audio_lyric, audio_only, lyric_only, audio_lyric, audio_lyric / len(pl_new)))
                    
MAX_LINE_LIMIT = 1000
# Load genre tags from LastFM
def load_lastfm_tag(dataset_root):
    # Init the class
    lf_api = lastFM.lastFM(dataset_root)
    # Check how many rows are waiting for genre tags
    # Note: Default value is NULL, which can be checked by where genre is null,
    # while updated value is json(null), which can only be checked out by where jsonlength(genre) <= 1.
    row_counter_waited = db.get_row_count_no_genre()
    # Start tagging
    pbar = tqdm(total=row_counter_waited)
    while(row_counter_waited > 0):
        # Read 1000 rows to be tagged
        track_ids, artist_names, song_titles = db.select_song_no_genre(line_limit=MAX_LINE_LIMIT)
        if track_ids is not None:
            for idx in range(len(track_ids)):
                genre_raw, genre_top = lf_api.get_tag(artist_names[idx], song_titles[idx])
                db.add_tag_to_song(track_ids[idx], json.dumps(genre_raw), json.dumps(genre_top))
                row_counter_waited -= 1
                pbar.update(1)
        else:
            # update row_counter_waited
            row_counter_waited = db.get_row_count_no_genre()
            pbar = tqdm(total=row_counter_waited)
        if row_counter_waited < MAX_LINE_LIMIT:
            # update row_counter_waited
            row_counter_waited = db.get_row_count_no_genre()
            pbar = tqdm(total=row_counter_waited)
    pbar.close()


# Load language tags from langdetect api
def load_language(dataset_root):
    # Init the lyric root
    lyric_root = dataset_root + lyricGenius.lyric_path
    # Check how many rows are waiting for language tags
    row_counter_waited = db.get_row_count_no_lang()
    # Start tagging
    pbar = tqdm(total=row_counter_waited)
    while(row_counter_waited > 0):
        track_ids = db.select_song_no_lang(line_limit=MAX_LINE_LIMIT)
        if track_ids is not None:
            for track_id in track_ids:
                try:
                    # check if it is a spotify id or a echo nest id
                    if not track_id.find("spotify") == -1:
                        _track_id = track_id.split(":")[2]
                    # find the lyric file
                    track_url = lyric_root + '/' + _track_id + '.txt'
                    # Get lyric string
                    lyric_file = open(track_url, encoding='utf-8')
                    lyric_str = lyric_file.read()
                    lang = detect(lyric_str)
                    # update database
                    db.add_lang_to_song(track_id, lang)
                    # update counter
                    row_counter_waited -= 1
                    pbar.update(1)
                except:
                    print('Cannot get language tag for track id: {}'.format(track_id))
                    # update lyric stat for solving the existing bug where lyric stat is 1 but no lyric file exists
                    db.update_lyric_stat(track_id)
        else:
            # update row_counter_waited
            row_counter_waited = db.get_row_count_no_lang()
            pbar = tqdm(total=row_counter_waited)
        if row_counter_waited < MAX_LINE_LIMIT:
            # update row_counter_waited
            row_counter_waited = db.get_row_count_no_lang()
            pbar = tqdm(total=row_counter_waited)
    pbar.close()


        
if __name__ == '__main__':
    
    # Init param
    # start_line = 0
    # start_doc = 0
    # start_pls = 0
    
    # load arguments
    # try:
    #     opts, args = getopt.getopt(sys.argv[1:], "")
    # except getopt.GetoptError:
    #     print("\
    #         -e load echo nest dataset\n\
    #         -s load spotify million playlist dataset\n\
    #         -m load million song link table for echo nest\n\
    #         -l start line (for '-e' option)\n\
    #         -d start document index (for '-s' option)\n\
    #         -p start playlist index (for '-s' option)")

    parser = argparse.ArgumentParser(description='Load Echo Nest/ Spotify Million Playlist Dataset to MySQL database')

    # Load dataset
    parser.add_argument('--echo', '-e', action='store_true', help='Load Echo Nest Dataset')
    parser.add_argument('--spotify', '-s', action='store_true', help='Load Spotify Million Playlist Dataset')
    parser.add_argument('--million', '-m', action='store_true', help='Load Million Song link table for Echo Nest')
    parser.add_argument('--line', '-l', type=int, default=0, help='Start line (for \'-e\' option)')
    parser.add_argument('--doc', '-d', type=int, default=0, help='Start document index (for \'-s\' option)')
    parser.add_argument('--plist', '-p', type=int, default=0, help='Start playlist index (for \'-s\' option)')
    parser.add_argument('--root', '-r', type=str, default='E:', help='Dataset root Path (the path where the dataset folder in)')
    parser.add_argument('--lastfm', '-f', action='store_true', help='Load LastFM tags from API')
    parser.add_argument('--lang', '-g', action='store_true', help='Load language tag from langdetect API ')

    # Analyze database
    parser.add_argument('--tailor_echo', '-t', action='store_true', help='Pre-process Echo Nest Dataset')

    args = parser.parse_args()
    
    msd_link_txt = args.root + '/' + msd_link_txt_n
    echo_nest_txt = args.root + '/' + echo_nest_txt_n
    spotify_path = args.root + '/' + spotify_path_n

    sp_api = spotifyApi.Spotify(args.root)
    ly_api = lyricGenius.LyricGenius(args.root)
    lf_api = lastFM.lastFM(args.root)

    if args.echo:
        print("Load Echo Nest Dataset from line {}.".format(args.line))
        load_echo_nest(echo_nest_txt, args.line)
    elif args.spotify:
        print("Load Spotify Dataset from document {}, playlist {}.".format(args.doc, args.plist))
        load_spotify_mpd(spotify_path, args.doc, args.plist)
    elif args.million:
        print("Load Million Song link table for Echo Nest Dataset")
        load_msd_link(msd_link_txt)
    elif args.lastfm:
        print("Load LastFM tags for Song Table")
        load_lastfm_tag(args.root)
    elif args.lang:
        print("Load language tag for Song Table")
        load_language(args.root)


    # Load the million song link table
    # load_msd_link()

    # Load echo nest interaction to echo nest table
    # load_echo_nest(329265)

    # Load spotify million playlist dataset
    # load_spotify_mpd(0, 41)

    exit
     