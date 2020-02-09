import os
import sys
sys.path.append(os.getcwd())
from HelperMarioGAN.helper import *
import toml

def generate_training_level():
    #print(os.getcwd())
    parsed_toml = toml.load("config/LevelConfig.tml")

    level_path=parsed_toml["LevelPath"]
    level_width=parsed_toml["LevelWidth"]
    compressed=parsed_toml["Compressed"]
    print("Generating training levels, each in width "+str(level_width))
    X, z_dims, index2str = get_windows_from_folder(level_path, level_width, compressed)
    print("Training Level Generation Finished")
    return X, z_dims, index2str

