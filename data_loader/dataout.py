# dataout.py
# Generate dataset json or txt file that can be used by our model from mysql database

import musicdb
from tqdm import tqdm
import os

# Init database
db = musicdb.MusicDB()

# Path to generated dataset
echo_nest_out_path = 'dataset/data'

