# musicdb.py
# API for mysql database, including inserting, selecting, updating database

import mysql.connector
import json
import api_key

hostname = "localhost"
user="root"
password=api_key.read_mysql_keys()

database_name = 'musicdb'

echo_nest_table_name = 'echonest'
echo_nest_filter_table = 'echofilter'
million_song_link_table_name = 'millionsonglink'
spotify_table_name = 'spotify'
song_table_name = 'songs'

class MusicDB(object):
    def __init__(self):
        # Check if the database exist
        self.create_database()
        # Init the cursor for database
        self.mydb = mysql.connector.connect(
            host=hostname,
            user=user,
            password=password,
            database=database_name
        )
        self.mycursor = self.mydb.cursor(buffered=True)
        # Check if the table exist
        self.create_tables()
        # For execute external mysql command
        self.echo_nest_table_name = echo_nest_table_name
        self.million_song_link_table_name = million_song_link_table_name
        self.spotify_table_name = spotify_table_name
        self.song_table_name = song_table_name
    
    def create_database(self):
        # Connect the database for the first time
        mydb = mysql.connector.connect(
        host=hostname,
        user=user,
        password=password
        )

        # Get the cursor
        mycursor = mydb.cursor()

        # Check if database exist
        is_database_exist = False
        mycursor.execute("SHOW DATABASES")
        for x in mycursor:
            #print(x[0])
            if x[0] == database_name:
                is_database_exist = True
                print("Database {} has already been existed.".format(database_name))
        
        # Create database
        if not is_database_exist:
            sql_cmd = "CREATE DATABASE " + database_name
            mycursor.execute(sql_cmd)

    def create_tables(self):
        # Create echonest table
        is_echo_nest_table_exist = False
        self.mycursor.execute("SHOW TABLES")
        for x in self.mycursor:
            if x[0] == echo_nest_table_name:
                is_echo_nest_table_exist = True
                print("Table {} has already been existed.".format(echo_nest_table_name))
        if not is_echo_nest_table_exist:
            sql_cmd = "CREATE TABLE {} (userid VARCHAR(255), playlist JSON, playlist_length INT, \
                none_audio_lyric INT, audio_only INT, lyric_only INT, audio_lyric INT, completeness FLOAT, \
                is_valid BOOLEAN DEFAULT 0)".format(echo_nest_table_name)
            self.mycursor.execute(sql_cmd)

        # Create echo nest filter table
        is_echo_nest_filter_exist = False
        self.mycursor.execute("SHOW TABLES")
        for x in self.mycursor:
            if x[0] == echo_nest_filter_table:
                is_echo_nest_filter_exist = True
                print("Table {} has already been existed.".format(echo_nest_filter_table))
        if not is_echo_nest_filter_exist:
            sql_cmd = "CREATE TABLE {} (userid VARCHAR(255), playlist JSON, playlist_length INT, \
                old INT, new INT)".format(echo_nest_filter_table)
            self.mycursor.execute(sql_cmd)

        # Create million song link table 
        # Note: track id here is for million song dataset, not the one we use for spotify
        is_million_link_table_exist = False
        self.mycursor.execute("SHOW TABLES")
        for x in self.mycursor:
            if x[0] == million_song_link_table_name:
                is_million_link_table_exist = True
                print("Table {} has already been existed.".format(million_song_link_table_name))
        if not is_million_link_table_exist:
            sql_cmd = "CREATE TABLE {} (trackid VARCHAR(255), songid VARCHAR(255), \
                artistname LONGTEXT, songtitle LONGTEXT)".format(million_song_link_table_name)
            self.mycursor.execute(sql_cmd)

        # Create spotify interaction table
        is_spotify_table_exist = False
        self.mycursor.execute("SHOW TABLES")
        for x in self.mycursor:
            if x[0] == spotify_table_name:
                is_spotify_table_exist = True
                print("Table {} has already been existed.".format(spotify_table_name))
        if not is_spotify_table_exist: # pid: playlist id
            sql_cmd = "CREATE TABLE {} (pid BIGINT, name LONGTEXT, playlist JSON, metainfo JSON, playlist_length INT,\
                none_audio_lyric INT, audio_only INT, lyric_only INT, audio_lyric INT, completeness FLOAT)"\
                .format(spotify_table_name)
            self.mycursor.execute(sql_cmd)

        # Create song table
        is_song_table_exist = False
        self.mycursor.execute("SHOW TABLES")
        for x in self.mycursor:
            if x[0] == song_table_name:
                is_song_table_exist = True
                print("Table {} has already been existed.".format(song_table_name))
        if not is_song_table_exist:
            sql_cmd = "CREATE TABLE {} (trackid VARCHAR(255), trackname LONGTEXT, \
                albumid VARCHAR(255), albumname LONGTEXT, artistid VARCHAR(255), \
                artistname LONGTEXT, release_date DATE, lang VARCHAR(255) DEFAULT NULL, \
                audio BOOLEAN, lyric BOOLEAN, echosongid VARCHAR(255), is_spotify BOOLEAN, \
                genre_top JSON DEFAULT NULL, genre_raw JSON DEFAULT NULL, preview_url LONGTEXT, \
                audiofeature JSON)".format(song_table_name)

            self.mycursor.execute(sql_cmd)

    # Execute external mysql query command
    def execute_cmd(self, sql_cmd):
        self.mycursor.execute(sql_cmd)
        return self.mycursor


    # Insert rows to million song link table
    # Input: val = [('track_id', 'song_id', 'artist_name', 'song_title'), (...), (...)]
    def insert_msd_link(self, val):
        sql_cmd = "INSERT INTO {} (trackid, songid, artistname, songtitle)"\
            .format(million_song_link_table_name)\
            + "VALUES (%s, %s, %s, %s)"
        self.mycursor.executemany(sql_cmd, val)
        self.mydb.commit()

    # Select row from million song link table according to song_id
    # Return: artist_name, song_title 
    def select_msd_link(self, song_id):
        sql_cmd = "SELECT * FROM {} WHERE songid='{}'"\
            .format(million_song_link_table_name, song_id)
        self.mycursor.execute(sql_cmd)
        # iterate the result: we select the first result here because there is no duplicate songid
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                # Deal with the bug from mysql connector
                if str(row[2]) == "inf":
                    row = list(row)
                    row[2] = "Infinity"
                    row = (*row, )
                if str(row[3]) == "inf":
                    row = list(row)
                    row[3] = "Infinity"
                    row = (*row, )
                return row[2], row[3]
        else:
            return None, None

    # Insert rows to song table
    # Input:
    # val = [(trackid, trackname, albumid, albumname, artistid, artistname, (all string)
    #      release_date (string: year-month-date), audio (0/1), lyric(0/1), 
    #      echosongid('None' or string), audiofeature(json)), (...), (...)]
    def insert_song(self, val):
        sql_cmd = "INSERT INTO {} (trackid, trackname, albumid, albumname, artistid, artistname, \
            release_date, audio, lyric, echosongid, is_spotify, preview_url, audiofeature)".format(song_table_name)\
            + "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.mycursor.executemany(sql_cmd, val)
        self.mydb.commit()
    
    # Select rows from song table according to echo nest song id
    # Input:
    # echo_song_id: song id from echo nest
    # Return:
    # success: song dictionary
    # fail: None
    def select_song(self, echo_song_id):
        sql_cmd = "SELECT * FROM {} WHERE echosongid='{}'"\
            .format(song_table_name, echo_song_id)
        self.mycursor.execute(sql_cmd)
        # iterate the result: we select the first result here because there is no duplicate songid
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                # Deal with the bug from mysql connector
                if str(row[1]) == "inf":
                    row = list(row)
                    row[1] = "Infinity"
                    row = (*row, )
                if str(row[3]) == "inf":
                    row = list(row)
                    row[3] = "Infinity"
                    row = (*row, )
                if str(row[5]) == "inf":
                    row = list(row)
                    row[5] = "Infinity"
                    row = (*row, )
                # Return the song dictionary
                return self.get_song_dict(row)
        else:
            return None

    # Select rows from song table according to spotify track id
    # Input:
    # track_id: spotify track id. e.g: spotify:track:1oMjdnSIwiQF0rnEUbCt2V
    # Return:
    # success: song dictionary
    # fail: None
    def select_song_spotify(self, track_id):
        sql_cmd = "SELECT * FROM {} WHERE trackid='{}'"\
            .format(song_table_name, track_id)
        self.mycursor.execute(sql_cmd)
        # iterate the result: we select the first result for there is no duplicate track id
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                # Deal with the bug from mysql connector
                if str(row[1]) == "inf":
                    row = list(row)
                    row[1] = "Infinity"
                    row = (*row, )
                if str(row[3]) == "inf":
                    row = list(row)
                    row[3] = "Infinity"
                    row = (*row, )
                if str(row[5]) == "inf":
                    row = list(row)
                    row[5] = "Infinity"
                    row = (*row, )
                # Return the song dictionary
                return self.get_song_dict(row)
        else:
            return None

    # Update echo song id in song table
    # Input:
    # track_id: spotify track id to select the row
    # echo_song_id: echo nest song id to be updated
    def update_echo_song_id(self, track_id, echo_song_id):
        sql_cmd = "UPDATE {} SET echosongid='{}' WHERE trackid='{}'"\
            .format(song_table_name, echo_song_id, track_id)
        self.mycursor.execute(sql_cmd)
        self.mydb.commit()
    
    # Update spotify status to 1 in song table
    # Input:
    # track_id: spotify track id to select the row
    def update_spotify_stat(self, track_id):
        sql_cmd = "UPDATE {} SET is_spotify='1' WHERE trackid='{}'"\
            .format(song_table_name, track_id)
        self.mycursor.execute(sql_cmd)
        self.mydb.commit()

    # Get the json/dict with song table columns
    def get_song_dict(self, row):
        song_dict = {
            "track_id": row[0],
            "track_name": row[1],
            "album_id": row[2],
            "album_name": row[3],
            "artist_id": row[4],
            "artist_name": row[5],
            "release_date": row[6].strftime("%Y-%m-%d") if row[6] is not None else None,
            "audio": row[7],
            "lyric": row[8],
            "echo_song_id": row[9],
            'is_spotify': row[10],
            "lang": row[11],
            "genre_top": row[12],
            "genre_raw": row[13],
            "preview_url": row[14],
            "audio_features": row[15],
        }
        return song_dict

    # Get val format for song table (with echo nest song id, for echo nest dataset)
    def get_song_val_echo(self, song_dict):
        val = (song_dict["track_id"], str(song_dict["track_name"]), song_dict["album_id"], str(song_dict["album_name"]), \
            song_dict["artist_id"], str(song_dict["artist_name"]), song_dict["release_date"], song_dict["audio"], \
                song_dict["lyric"], song_dict["echo_song_id"], '0', song_dict["preview_url"], json.dumps(song_dict["audio_features"]))
        return val

    # Get val format for song table (without echo nest song id, for spotify dataset)
    def get_song_val(self, song_dict):
        val = (song_dict["track_id"], str(song_dict["track_name"]), song_dict["album_id"], str(song_dict["album_name"]), \
            song_dict["artist_id"], str(song_dict["artist_name"]), song_dict["release_date"], song_dict["audio"], \
                song_dict["lyric"], '0', '1', song_dict["preview_url"], json.dumps(song_dict["audio_features"]))
        return val
        
    # Insert row to echo nest table
    def insert_echo_nest(self, val):
        sql_cmd = "INSERT INTO {} (userid, playlist, playlist_length, \
            none_audio_lyric, audio_only, lyric_only, audio_lyric, completeness)".format(echo_nest_table_name)\
            + " VALUE (%s, %s, %s, %s, %s, %s, %s, %s)"
        self.mycursor.execute(sql_cmd, val)
        self.mydb.commit()

    # Select rows from echo nest table by user id
    def select_echo_nest(self, user_id):
        sql_cmd = "SELECT * FROM {} WHERE userid='{}'".format(echo_nest_table_name, user_id)
        self.mycursor.execute(sql_cmd)
        # iterate the result: we select the first result here because there is no duplicate songid
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                # Return the song dictionary and list of counters
                return json.loads(row[1]), [row[2], row[3], row[4], row[5], row[6]]
        else:
            return None, None
    
    # Insert row to spotify table
    def insert_spotify(self, val):
        sql_cmd = "INSERT INTO {} (pid, name, playlist, metainfo, playlist_length, \
            none_audio_lyric, audio_only, lyric_only, audio_lyric, completeness)".format(spotify_table_name)\
            + " VALUE (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.mycursor.execute(sql_cmd, val)
        self.mydb.commit()

    # Select rows from spotify table by pid (playlist id)
    # Return:
    # success: playlist name for checking
    # fail: None
    def select_spotify(self, pid):
        sql_cmd = "SELECT * FROM {} WHERE pid='{}'".format(spotify_table_name, pid)
        self.mycursor.execute(sql_cmd)
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                return row[1]
        else:
            return None

    # Select rows without genre tag from song table
    # Return:
    # success: track_ids [], artist_names [], song_titles []
    # fail: None, None, None
    def select_song_no_genre(self, line_limit=1000):
        sql_cmd = "SELECT * FROM {} WHERE genre_raw IS NULL OR genre_top IS NULL"\
            .format(song_table_name) + " LIMIT 0, {}".format(line_limit)
        self.mycursor.execute(sql_cmd)
        track_ids = []
        artist_names = []
        song_titles = []
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                track_ids.append(row[0])
                artist_names.append(row[5])
                song_titles.append(row[1])
        else:
            return None, None, None
        return track_ids, artist_names, song_titles

    # Get row count for songs without genre tags
    # Return:
    # success: int(row number)
    # fail: None
    def get_row_count_no_genre(self):
        sql_cmd = "SELECT COUNT(*) FROM {} WHERE genre_raw IS NULL OR genre_top IS NULL"\
            .format(song_table_name)
        self.mycursor.execute(sql_cmd)
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                return row[0]
        else:
            return None

    # Add tags to existing rows: genre_raw, genre_top have to be json
    def add_tag_to_song(self, track_id, genre_raw, genre_top):
        sql_cmd = "UPDATE {} SET genre_raw='{}', genre_top='{}' WHERE trackid='{}'"\
            .format(song_table_name, genre_raw, genre_top, track_id)
        self.mycursor.execute(sql_cmd)
        self.mydb.commit()

    # Select rows without language tag from song table
    # Return:
    # success: track_ids []
    # fail: None
    def select_song_no_lang(self, line_limit=1000):
        sql_cmd = "SELECT * FROM {} WHERE lang IS NULL and lyric=1"\
            .format(song_table_name) + " LIMIT 0, {}".format(line_limit)
        self.mycursor.execute(sql_cmd)
        track_ids = []
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                track_ids.append(row[0])
        else:
            return None
        return track_ids

    # Get row count for songs without language tags
    # Return:
    # success: int (row counter0)
    # fail: None
    def get_row_count_no_lang(self):
        sql_cmd = "SELECT COUNT(*) FROM {} WHERE lang IS NULL and lyric=1"\
            .format(song_table_name)
        self.mycursor.execute(sql_cmd)
        if self.mycursor.rowcount > 0:
            for row in self.mycursor:
                return row[0]
        else:
            return None

    # Update language in song table
    # Input:
    # track_id: spotify track id to select the row
    # language (str): language sign from langdetect api
    def add_lang_to_song(self, track_id, lang):
        sql_cmd = "UPDATE {} SET lang='{}' WHERE trackid='{}'"\
            .format(song_table_name, lang, track_id)
        self.mycursor.execute(sql_cmd)
        self.mydb.commit()

    # Update lyric stat in song table (for solving a bug)
    def update_lyric_stat(self, track_id):
        sql_cmd = "UPDATE {} SET lyric=0 WHERE trackid='{}'"\
            .format(song_table_name, track_id)
        self.mycursor.execute(sql_cmd)
        self.mydb.commit()

    def update_audio_lyric_stat(self, track_id, audio, lyric):
        sql_cmd = "UPDATE {} SET audio='{}', lyric='{}' WHERE trackid='{}'"\
            .format(song_table_name, audio, lyric, track_id)
        self.mycursor.execute(sql_cmd)
        self.mydb.commit()

    # Read rows from echo nest table
    # Input:
    # start: start index
    # length: number of rows to read
    # Return:
    # success: [(user_id, playlist json, playlist_length), (...), ...] list of tuples
    # fail: None
    def read_echo_nest(self, start, length):
        sql_cmd = "SELECT * FROM {} LIMIT {}, {}"\
            .format(echo_nest_table_name, start, start + length)
        self.mycursor.execute(sql_cmd)
        if self.mycursor.rowcount > 0:
            res = []
            for row in self.mycursor:
                res.append({
                    'user_id': row[0],
                    'playlist': row[1],
                    'playlist_length': row[2],
                    "audio_lyric": row[6],
                    "completeness": row[7],
                    "is_valid": row[8]
                })
            return res
        else:
            return None

    def update_echo_nest(self, user_id, playlist, audio_lyric, completeness, is_valid):
        sql_cmd = "UPDATE {} ".format(echo_nest_table_name) + \
         "SET playlist=%s, audio_lyric=%s, completeness=%s, is_valid=1 WHERE userid=%s"
        val = (playlist, audio_lyric, completeness, user_id)
        self.mycursor.execute(sql_cmd, val)
        self.mydb.commit()

    # Insert and read echo nest filter table (for generating dataset and sub-dataset)
    def insert_echo_nest_filter(self, val):
        sql_cmd = "INSERT INTO {} (userid, playlist, playlist_length, old, new\
            )".format(echo_nest_filter_table)\
            + " VALUE (%s, %s, %s, %s, %s)"
        self.mycursor.execute(sql_cmd, val)
        self.mydb.commit()

    def read_echo_nest_filter(self, start, length):
        sql_cmd = "SELECT * FROM {} LIMIT {}, {}"\
            .format(echo_nest_filter_table, start, start + length)
        self.mycursor.execute(sql_cmd)
        if self.mycursor.rowcount > 0:
            res = []
            for row in self.mycursor:
                res.append({
                    'user_id': row[0],
                    'playlist': row[1],
                    'playlist_length': row[2],
                    "old": row[3],
                    "new": row[4]
                })
        else:
            return None

    def check_if_user_exists_filter(self, user_id):
        sql_cmd = "SELECT * FROM {}".format(echo_nest_filter_table)\
            + " WHERE userid='{}'".format(user_id)
        self.mycursor.execute(sql_cmd)
        if self.mycursor.rowcount > 0:
            return True
        else:
            return False

        