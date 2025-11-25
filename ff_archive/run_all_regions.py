# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:14:26 2024

@author: EagleView
"""

# # Input text for training
# text_tiles = """
# SouthAmerica : 10N_080W, 10N_070W, 10N_060W, 00N_080W, 00N_070W, 00N_060W, 00N_050W, 10S_080W, 10S_070W, 10S_060W
# Africa : 10N_000E, 10N_010E, 10N_020E, 00N_000E, 00N_010E, 00N_020E
# Asia : 10N_090E, 10N_100E, 00N_100E, 10N_110E, 00N_110E, 10N_120E, 00N_120E, 00N_130E, 00N_140E
# """


# Input text for predictions
text_tiles = """
SouthAmerica : 10N_080W,10N_070W,10N_060W,00N_080W,00N_070W,00N_060W,00N_050W,10S_080W,10S_070W,10S_060W,00N_090W,10S_050W  
Africa : 10N_000E, 10N_010E, 10N_020E, 00N_000E, 00N_010E, 00N_020E, 10N_030E
Asia : 10N_090E, 10N_100E, 00N_100E, 10N_110E, 00N_110E, 10N_120E, 00N_120E, 00N_130E, 00N_140E, 00N_090E, 10S_140E, 10S_150E, 00N_150E
"""

# # Input text for predictions
# text_tiles = """
# Asia : 10N_090E, 10N_100E, 00N_100E, 10N_110E, 00N_110E, 10N_120E, 00N_120E, 00N_130E, 00N_140E, 00N_090E, 10S_140E, 10S_150E, 00N_150E
# """

# asi with laos and cambodia: Asia : 20N_100E, 30N_100E, 10N_090E, 10N_100E, 00N_100E, 10N_110E, 00N_110E, 10N_120E, 00N_120E, 00N_130E, 00N_140E, 00N_090E, 10S_140E, 10S_150E, 00N_150E

# Input text
text_shape = """
SouthAmerica : C:/PostDoc/deforestation_project/Laura_CNN_experiments/south_america_shape.shp, all, SouthAmerica
Africa : C:/PostDoc/deforestation_project/Laura_CNN_experiments/africa_shape.shp, all, Africa
Asia : C:/PostDoc/deforestation_project/Laura_CNN_experiments/asia_shape.shp, all, Asia
"""

# # # Input text
# text_tiles = """
# Asia : 10N_090E, 10N_100E, 00N_100E, 10N_110E, 00N_110E, 10N_120E, 00N_120E, 00N_130E, 00N_140E
# """

# # # Input text
# text_shape = """
# Asia : C:/PostDoc/deforestation_project/Laura_CNN_experiments/asia_shape.shp, all, Asia
# """

# Parsing the text into a dictionary
region_dict = {}

for line in text_tiles.strip().split("\n"):
    # Split the line into region name and tiles
    region, tiles = line.split(" : ")
    # Convert the list of tiles into a Python list
    tile_list = [tile.strip() for tile in tiles.split(",")]
    # Add the region and its tiles to the dictionary
    region_dict[region.strip()] = [tile_list]
    
for line in text_shape.strip().split("\n"):
    # Split the line into region name and tiles
    region, info = line.split(" : ")
    shape, column, country = info.split(",")

    region_dict[region.strip()].append([shape.strip()])
    region_dict[region.strip()].append([column.strip()])
    region_dict[region.strip()].append([country.strip()])
    

import subprocess


# Path to the script to be called
script_prec = "organize_data_normalize.py"
script_mask = "create_country_masks.py"
script_train = "main_multi_tiles.py"
script_eval = "evaluation_multi_tiles.py"
script_metric = "metrics_fixed_tresh_multi_tiles.py"
secript_merge = "merge_tiles.py"
secript_xgboost = "xgboost_train_pred.py"
script_captum = "xai_captum.py"

# Iterate over regions and tiles
for region, info in region_dict.items():
    tiles, shape, column, country = info
    tile_str = ','.join(tiles)
    print(f"Processing region: {region} with tiles: {tile_str}")

    # # Construct the command
    # command = [
    #     "python", script_prec,
    #     "--tiles", tile_str,
    # ]

    # # # #Execute the script
    # subprocess.run(command)
    
    # # Construct the command
    # command = [
    #     "python", script_mask,
    #     "--tiles", tile_str,
    #     "--shapefile", shape[0],
    #     "--column", column[0],
    #     "--country", country[0],
    # ]

    # #Execute the script
    # subprocess.run(command)
    
    # # #Construct the command
    # command = [
    #     "python", script_train,
    #     "--tiles", tile_str,
    #     "--dump_path", f"./exp_tiles_continent_v2/{region}",
    # ]
 
    # # # Execute the script
    # subprocess.run(command) 
    
    # Construct the command
    # command = [
    #     "python", secript_xgboost,
    #     "--tiles", tile_str,
    #     "--dump_path", f"./exp_tiles_continent_xgboost_v2/{region}",
    # ]
 
    # # # Execute the script
    # subprocess.run(command) 
    
    # command = [
    #     "python", script_eval,
    #     "--tiles", tile_str,
    #     "--dump_path", f"./exp_tiles_continent_server/{region}",
    # ]

    # # Execute the script
    # subprocess.run(command)
    
    # command = [
    #     "python", script_metric,
    #     "--tiles", tile_str,
    #     "--pred_path", f"./exp_tiles_continent/{region}",
    # ]

    # # Execute the script
    # subprocess.run(command)

    # command = [
    #     "python", secript_merge,
    #     "--tiles", tile_str,
    #     "--shapefile", shape[0],
    #     "--dump_path", f"./exp_tiles_continent_xgboost_v2/{region}",
    # ]

    # # Execute the script
    # subprocess.run(command)
    
    command = [
        "python", script_captum,
        "--tiles", tile_str,
        "--shapefile", shape[0],
        "--dump_path", f"./exp_tiles_continent_server/{region}",
    ]

    # Execute the script
    subprocess.run(command)




