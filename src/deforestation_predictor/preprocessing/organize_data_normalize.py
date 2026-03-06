# -*- coding: utf-8 -*-
"""
Script Name: Data Split and Normalization for Deforestation Analysis
Created on: Wed Feb 7 12:35:14 2024

Author: Laura Elena Cue La Rosa
Project: WUR-WWF Deforestation Project

Description:
    This script is designed to prepare satellite imagery data for machine learning analysis,
    focusing on the detection and prediction of deforestation events. It splits the data into
    training, testing, and validation sets and applies normalization based on predefined
    maximum values.

    The script handles various types of satellite-derived variables, including dynamic,
    semi-dynamic, and static variables, and organizes them accordingly for the analysis.

"""


import argparse
import math
import os
from src.utils.logger import create_logger

import numpy as np
import glob
from pathlib import Path
from datetime import datetime

from src.utils.preprocessing_helpers import (
    read_tiff,
    check_folder,
)


def main(parser):
    global args
    args = parser.parse_args()
    check_folder(args.dump_path)
    
    args.tiles = args.tiles.split(",")  # Convert the comma-separated string into a list
    
    logger = create_logger(os.path.join(args.dump_path, "split_and_norm.log"),rank=0)
    logger.info("============ Initialized logger ============")
    
    for tile in args.tiles:
        
        if os.path.isfile(os.path.join(args.dump_path,'{}_gt_tr.npy'.format(tile))):
             logger.info(f"Tile {tile} already processed")
             continue
        else:

            ######### Organazie data and split train and test ############
            dynam_var_train, gt_stack_tr, dynam_var_val, gt_stack_val, dynam_var_test, gt_stack_test, mask = prepare_data(tile,logger)
            
            # filter nana and inf data
            dynam_var_train[np.isnan(dynam_var_train)] = 0
            dynam_var_val[np.isnan(dynam_var_val)] = 0
            dynam_var_test[np.isnan(dynam_var_test)] = 0
            
            dynam_var_train[np.isinf(dynam_var_train)] = 0
            dynam_var_val[np.isinf(dynam_var_val)] = 0
            dynam_var_test[np.isinf(dynam_var_test)] = 0
            
            # convert to uint8 
            dynam_var_train*=255
            dynam_var_val*=255
            dynam_var_test*=255
            
            np.save(os.path.join(args.dump_path,'{}_var_train'.format(tile)),dynam_var_train.astype('uint8'))
            np.save(os.path.join(args.dump_path,'{}_gt_tr'.format(tile)),gt_stack_tr)
            
            np.save(os.path.join(args.dump_path,'{}_var_val'.format(tile)),dynam_var_val.astype('uint8'))
            np.save(os.path.join(args.dump_path,'{}_gt_val'.format(tile)),gt_stack_val)
            
            np.save(os.path.join(args.dump_path,'{}_var_test'.format(tile)),dynam_var_test.astype('uint8'))
            np.save(os.path.join(args.dump_path,'{}_gt_test'.format(tile)),gt_stack_test)
            
            # np.save(os.path.join(args.dump_path,'{}_var_train'.format(tile)),dynam_var_train.astype('float16'))
            # np.save(os.path.join(args.dump_path,'{}_gt_tr'.format(tile)),gt_stack_tr)
            
            # np.save(os.path.join(args.dump_path,'{}_var_val'.format(tile)),dynam_var_val.astype('float16'))
            # np.save(os.path.join(args.dump_path,'{}_gt_val'.format(tile)),gt_stack_val)
            
            # np.save(os.path.join(args.dump_path,'{}_var_test'.format(tile)),dynam_var_test.astype('float16'))
            # np.save(os.path.join(args.dump_path,'{}_gt_test'.format(tile)),gt_stack_test)
            
            np.save(os.path.join(args.dump_path,'{}_mask'.format(tile)),mask)
            
            del dynam_var_train, dynam_var_val, dynam_var_test
            del gt_stack_tr, gt_stack_val, gt_stack_test
            del mask
    
    
    
# Define a function to extract the date from the filename
def extract_date(filename):
    # Assuming the filename has the format 'YYYY-M-DD'
    parts = Path(filename).stem.split('_')[-2].split('-')
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    return datetime(year, month, day)
    

        
def prepare_data(tile, logger):
    # define empty variables 
    dynam_var_train = []
    dynam_var_val = []
    dynam_var_test = []
    
    list_year_tr = []
    list_year_val = []
    list_year_test = []
    
    gt_tr = []
    gt_val = []
    gt_test = []
    
    
    list_year_ind = False
    
    # define path for the specific tile
    data_gt = os.path.join(args.data_gt,tile)
    data_input = os.path.join(args.data_input,tile)
    
    gt_list = glob.glob(data_gt + f'/*{args.ref_name}*.tif')
#    print(gt_list)
    # Sort the image files based on the extracted date
    sorted_gt_files = sorted(gt_list, key=extract_date)
    
    #### get train and test for dynamic vars
    for var, max_val in args.dynamic_var:
        img_list = glob.glob(data_input + '/*{}*'.format(var) + '*.tif')
        
        if len(img_list) > 0:
        
            # Sort the image files based on the extracted date
            sorted_image_files = sorted(img_list, key=extract_date)
            
            # Initialize two empty lists to hold the split file paths
            train_split = []
            test_split = []
            val_split = []
            
            # Iterate through each file path in the list
            for cont in range(len(sorted_gt_files)):
                # Extract the date part of the filename and format it as YYYY-MM-DD
                # Assuming the date is always in the format '20N_100E_YYYY-MM-DD_confidence.tif'
                date_str = Path(sorted_gt_files[cont]).stem.split('_')[2]
                 
                try:
                    path = [x for x in sorted_image_files if date_str in x][0]
                except:
                    previous_date = Path(sorted_gt_files[cont-1]).stem.split('_')[2]
                    logger.info(f'{var} date {date_str} not avaliable, taking previous date {previous_date}')  
    
                # Compare the extracted date with the split date and categorize the file path accordingly
                if date_str >= args.startdate_train and date_str <= args.enddate_train:
                    train_split.append(path)
                    if not list_year_ind:
                        list_year_tr.append(date_str[:4]) 
                        gt_tr.append([x for x in sorted_gt_files if Path(x).stem.split('_')[2] == date_str][0])
                        
                if date_str >= args.startdate_val and date_str <= args.enddate_val:
                    val_split.append(path)
                    if not list_year_ind:
                        list_year_val.append(date_str[:4]) 
                        gt_val.append([x for x in sorted_gt_files if Path(x).stem.split('_')[2] == date_str][0])
                        
                if date_str >= args.startdate_test and date_str <= args.enddate_test:
                    test_split.append(path)
                    if not list_year_ind:
                        list_year_test.append(date_str[:4]) 
                        gt_test.append([x for x in sorted_gt_files if Path(x).stem.split('_')[2] == date_str][0])
                        
            # Define maximum value for normalization (divide by maximum if args.normalize set to TRUE el divide by 1)
            if args.normalize:
                maximum = 1 if float(max_val) == 0 else float(max_val)
            else:
                maximum = 1
            
            # stack data and normalize   
            train_stack = np.stack([read_tiff(x).astype('float32') for x in train_split],axis=0)
            logger.info(f'{var} stack shape {train_stack.shape} with max val {np.max(train_stack[~np.isnan(train_stack)])} and min val {np.min(train_stack[~np.isnan(train_stack)])}')  
            
            train_stack/=maximum
            val_stack = np.stack([read_tiff(x).astype('float32') for x in val_split],axis=0)/maximum

            test_stack = np.stack([read_tiff(x).astype('float32') for x in test_split],axis=0)/maximum   
            
            list_year_ind = True
            
            # append feature to the dynamic vars
            dynam_var_train.append(train_stack)
            dynam_var_val.append(val_stack)
            dynam_var_test.append(test_stack)

        else:
            logger.info(f'No {var} feature in tile')  
            
    
    train_dates = [Path(p).stem.split('_')[2] for p in gt_tr]
    val_dates = [Path(p).stem.split('_')[2] for p in gt_val]
    test_dates = [Path(p).stem.split('_')[2] for p in gt_test]

    # Log the exact timesteps collected for each split
    logger.info(f"=== Temporal Sequence Selected for Tile {tile} ===")
    logger.info(f"Train timesteps ({len(train_dates)}): {train_dates}")
    logger.info(f"Validation timesteps ({len(val_dates)}): {val_dates}")
    logger.info(f"Test timesteps ({len(test_dates)}): {test_dates}")
    logger.info("==================================================")   
    
    # stack all dynamic features
    dynam_var_train = np.stack(dynam_var_train,axis=0)
    dynam_var_val = np.stack(dynam_var_val,axis=0)
    dynam_var_test = np.stack(dynam_var_test,axis=0)
    
    
    #TODO: this was set becuase some features have values larger thatn maximum?
    if args.normalize:
        dynam_var_train[dynam_var_train>1] = 0
        dynam_var_val[dynam_var_val>1] = 0
        dynam_var_test[dynam_var_test>1] = 0
      
    
    #### get train, validation and test for autogenrated dynamic vars
    for var, max_val in args.auto_gen_dynamic:
        if args.normalize:
            maximum = 1 if float(max_val) == 0 else float(max_val)
        else:
            maximum = 1
            
        if var == 'month':  
            train_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*int(x.split('_')[-2].split('-')[1]) for x in train_split],axis=0).astype('float16')/maximum
            val_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*int(x.split('_')[-2].split('-')[1]) for x in val_split],axis=0).astype('float16')/maximum
            test_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*int(x.split('_')[-2].split('-')[1]) for x in test_split],axis=0).astype('float16')/maximum
            
        elif var == 'sinmonth':
            train_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*math.sin(int(x.split('_')[-2].split('-')[1])) for x in train_split],axis=0).astype('float16')/maximum
            val_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*math.sin(int(x.split('_')[-2].split('-')[1])) for x in val_split],axis=0).astype('float16')/maximum
            test_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*math.sin(int(x.split('_')[-2].split('-')[1])) for x in test_split],axis=0).astype('float16')/maximum

        elif var == 'monthssince2019':
            train_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*months_since_january_2019(Path(x).stem.split('_')[2][2:-3]) for x in train_split],axis=0).astype('float16')/maximum
            val_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*months_since_january_2019(Path(x).stem.split('_')[2][2:-3]) for x in val_split],axis=0).astype('float16')/maximum
            test_stack = np.stack([np.ones((train_stack.shape[1],train_stack.shape[2]))*months_since_january_2019(Path(x).stem.split('_')[2][2:-3]) for x in test_split],axis=0).astype('float16')/maximum


        # append feature to the dynamic vars
        dynam_var_train = np.concatenate((dynam_var_train, train_stack[np.newaxis, :, :, :]), axis=0)
        dynam_var_val = np.concatenate((dynam_var_val, val_stack[np.newaxis, :, :, :]), axis=0)
        dynam_var_test = np.concatenate((dynam_var_test, test_stack[np.newaxis, :, :, :]), axis=0)
    
    #### get train, validation and test for yearly vars and normalize if TRUE 
    # concatenate withg the dynamic var
    # train
    for var, max_val in args.yearly_var:
        yearly_var = []
        for yr in list_year_tr:
            try:
                img = read_tiff(glob.glob(data_input + '/*{}*{}*'.format(yr,var) + '*.tif')[0]).astype('float32')
            except:
                logger.info(f'No data found for {var} for year {yr}. Taking previous year')
                img = read_tiff(glob.glob(data_input + '/*{}*{}*'.format(int(yr)-1,var) + '*.tif')[0]).astype('float32')

            yearly_var.append(img)
        
        yearly_var = np.stack(yearly_var,axis=0)
        
        logger.info(f'{var} stack shape {yearly_var.shape} with max val {np.max(yearly_var[~np.isnan(yearly_var)])} and min val {np.min(yearly_var[~np.isnan(yearly_var)])}')  
        
        if args.normalize:
            maximum = 1 if float(max_val) == 0 else float(max_val)
        else:
            maximum = 1
            
        yearly_var/= maximum
        
        yearly_var = yearly_var[np.newaxis, :, :, :]
            
        dynam_var_train = np.concatenate((dynam_var_train,yearly_var), axis=0)
        
    # validation    
    for var, max_val in args.yearly_var:
        yearly_var = []
        for yr in list_year_val:
            try:
                img = read_tiff(glob.glob(data_input + '/*{}*{}*'.format(yr,var) + '*.tif')[0]).astype('float32')
            except:
                logger.info(f'No data found for {var} for year {yr}. Taking previous year')
                img = read_tiff(glob.glob(data_input + '/*{}*{}*'.format(int(yr)-1,var) + '*.tif')[0]).astype('float32')
            
            yearly_var.append(img)
        
        yearly_var = np.stack(yearly_var,axis=0)
        
        if args.normalize:
            maximum = 1 if float(max_val) == 0 else float(max_val)
        else:
            maximum = 1
            
        yearly_var/= maximum
        
        yearly_var = yearly_var[np.newaxis, :, :, :]
            
        dynam_var_val = np.concatenate((dynam_var_val,yearly_var), axis=0)
        
        
    # test    
    for var, max_val in args.yearly_var:
        yearly_var = []
        for yr in list_year_test:
            
            # Search backwards from the target year to find the most recent available file
            found_file = None
            for current_yr in range(int(yr), 1990, -1):
                files = glob.glob(data_input + '/*{}*{}*'.format(current_yr, var) + '*.tif')
                if files:
                    found_file = files[0]
                    if current_yr != int(yr):
                        logger.info(f'No data found for {var} for year {yr}. Taking data from year {current_yr} instead.')
                    break
            
            if found_file:
                img = read_tiff(found_file).astype('float32')
            else:
                # Raise an explicit error if no historical data is found at all
                raise FileNotFoundError(f"Missing data for {var} for year {yr} and all preceding years.")
            
            yearly_var.append(img)
        
        yearly_var = np.stack(yearly_var,axis=0)
        
        # Normalize and append to the test stack
        if args.normalize:
            maximum = 1 if float(max_val) == 0 else float(max_val)
        else:
            maximum = 1
            
        yearly_var /= maximum
        
        yearly_var = yearly_var[np.newaxis, :, :, :]
        dynam_var_test = np.concatenate((dynam_var_test, yearly_var), axis=0)
        
    
    #### get train, validation and test static vars
    if len(args.static_names)>0:
        static_var = []
    
        for var, max_val in args.static_names:
            if var == 'xy':
                longlat = gfw_tile_to_coordinates(tile)
                coordx, coordy = generate_geo_coordinates_array(longlat,dynam_var_train.shape[2])
                coordx, coordy = normalize_geo_coordinates(coordx, coordy)
                
                static_var.append(coordx.astype('float32'))
                static_var.append(coordy.astype('float32'))
                
            
            else:
                img = read_tiff(glob.glob(data_input + '/*{}*'.format(var) + '*.tif')[0]).astype('float32')
                logger.info(f'{var} stack shape {img.shape} with max val {np.max(img[~np.isnan(img)])} and min val {np.min(img[~np.isnan(img)])}')  
                
                if args.normalize or  var == 'populationcurrent' or var=='populationincrease':
                    maximum = 1 if float(max_val) == 0 else float(max_val)
                else:
                    maximum = 1
                    
                img/= maximum
                
                static_var.append(img)
         
            
        # concatenate with dynamic vars 
        static_var = np.stack(static_var,axis=0)
        static_var = static_var[:, np.newaxis, :, :]
        
        static_var_tr = np.repeat(static_var, dynam_var_train.shape[1], axis=1)
        dynam_var_train = np.concatenate((dynam_var_train,static_var_tr), axis=0)
        
        static_var_val = np.repeat(static_var, dynam_var_val.shape[1], axis=1)
        dynam_var_val = np.concatenate((dynam_var_val,static_var_val), axis=0)
        
        static_var_test = np.repeat(static_var, dynam_var_test.shape[1], axis=1)
        dynam_var_test = np.concatenate((dynam_var_test,static_var_test), axis=0)
    
    
    # stack groundtruth    
    gt_stack_tr = np.stack([read_tiff(x) for x in gt_tr],axis=0)
    gt_stack_tr[gt_stack_tr>0] = 1
    
    gt_stack_val = np.stack([read_tiff(x) for x in gt_val],axis=0)
    gt_stack_val[gt_stack_val>0] = 1
    
    gt_stack_test = np.stack([read_tiff(x) for x in gt_test],axis=0)
    gt_stack_test[gt_stack_test>0] = 1
    
    
    # define region of interest using the defined feature mask
    mask = read_tiff(glob.glob(data_input + '/*_{}'.format(args.mask_name) + '*.tif')[0]).astype('float32')
    mask[np.isnan(mask)] = 0
    mask[mask>0] = 1
    mask = mask.astype("uint8")
    
    # set value to mask-out regions of sea
    gt_stack_tr[:,mask==0] = 2
    gt_stack_tr[:,np.isnan(mask)] = 2
    
    gt_stack_val[:,mask==0] = 2
    gt_stack_val[:,np.isnan(mask)] = 2
    
    gt_stack_test[:,mask==0] = 2
    gt_stack_test[:,np.isnan(mask)] = 2
    
        
    return dynam_var_train, gt_stack_tr, dynam_var_val, gt_stack_val, dynam_var_test, gt_stack_test, mask

def gfw_tile_to_coordinates(tile_name):
    """
    Convert a GFW tile name to its bounding box coordinates.

    Parameters:
    - tile_name: The tile name in the format "00N_010E"

    Returns:
    - A dictionary with the bounding box coordinates: {south, west, north, east}
    """
    # Parse latitude and longitude from tile name
    lat_dir, lon_str = tile_name.split("_")
    lat = int(lat_dir[:-1])
    lon = int(lon_str[:-1])

    # Adjust latitude based on direction (N or S)
    if 'S' in lat_dir:
        south = - lat - 10
        north = -lat
    else:  # 'N' in lat_dir
        south = lat - 10
        north = lat

    # Adjust longitude based on direction (E or W)
    if 'W' in lon_str:
        west = - lon
        east = - lon + 10
    else:  # 'E' in lon_str
        west = lon
        east = lon + 10

    return {'south': south, 'west': west, 'north': north, 'east': east}

def normalize_geo_coordinates(lon_array, lat_array):
    """
    Normalize the geographic longitude and latitude coordinates between -1 and 1.
    
    Parameters:
    - lon_array: NumPy array of longitude coordinates.
    - lat_array: NumPy array of latitude coordinates.
    
    Returns:
    - Two NumPy arrays: normalized longitude and latitude coordinates.
    """
    # Normalize longitude and latitude
    normalized_lon_array = lon_array / 180.0
    normalized_lat_array = lat_array / 90.0
    
    return normalized_lon_array, normalized_lat_array


def generate_geo_coordinates_array(tile_bounds, tile_size):
    """
    Generate NumPy arrays with the geographic longitude and latitude coordinates for all pixels within a tile.
    
    Parameters:
    - tile_bounds: The geographic bounds of the tile in degrees (south, west, north, east).
    - tile_size: The size of the tile in pixels (assuming square tile).
    
    Returns:
    - Two NumPy arrays: one for longitude and one for latitude coordinates of all pixels.
    """
    # Unpack the tile bounds
    south, west, north, east = tile_bounds.values()
    
    # Create arrays for longitude and latitude
    lon_array = np.zeros((tile_size, tile_size))
    lat_array = np.zeros((tile_size, tile_size))
    
    # Calculate the span each pixel represents
    lon_per_pixel = (east - west) / tile_size
    lat_per_pixel = (north - south) / tile_size
    
    # Fill the arrays with geographic coordinates
    for y in range(tile_size):
        for x in range(tile_size):
            longitude = west + (x + 0.5) * lon_per_pixel
            latitude = north - (y + 0.5) * lat_per_pixel  # Subtracting because y increases down the image
            
            lon_array[y, x] = longitude
            lat_array[y, x] = latitude
    
    return lon_array, lat_array


def months_since_january_2019(date_str):
    # Parse the month and year from the string
    year, month = map(int, date_str.split('-'))
    year += 2000  # Assuming the year is in the 2000s

    # Create a datetime object for the parsed date
    date = datetime(year, month, 1)

    # Create a datetime object for January 2019
    start_date = datetime(2019, 1, 1)

    # Calculate the difference in months
    delta = (date.year - start_date.year) * 12 + date.month - start_date.month

    return delta



if __name__ == '__main__':
    """
    Script for Preparing Satellite Imagery Data Sets for Deforestation Analysis

    Optional arguments:
      -h, --help              Show this help message and exit
      --tile                  Tile to be processed
      --dump_path             Path where processed files will be saved
      --data_input            Path containing the input variables
      --data_gt               Path containing the groundtruth data
      --dynamic_var           List of dynamic variables and their max values for normalization
      --auto_gen_dynamic      List of auto-generated dynamic variables and their max values for normalization
      --yearly_var            List of semi-dynamic variables and their max values for normalization
      --static_names          List of static variables and their max values for normalization
      --ref_name              Reference variable for groundtruth
      --mask_name             Variable used to draw the region's mask
      --normalize             Normalize data by maximum values (default: True)
      --enddate_train         End date for the training data set
      --startdate_val         Start date for the validation data set
      --enddate_val           End date for the validation data set
      --startdate_test        Start date for the test data set
      --enddate_test          End date for the test data set

    Usage:
        > python this_script.py [-h] [--tile STR] [--dump_path STR] [--data_input STR] [--data_gt STR]
                                [--dynamic_var LIST] [--auto_gen_dynamic LIST] [--yearly_var LIST]
                                [--static_names LIST] [--ref_name STR] [--mask_name STR]
                                [--normalize BOOL] [--enddate_train DATE] [--startdate_val DATE]
                                [--enddate_val DATE] [--startdate_test DATE] [--enddate_test DATE]

    Example:
        > python this_script.py --tile 10S_070W --dump_path ./data_npy_normby_wwf
                                --data_input /path/to/input --data_gt /path/to/gt --normalize True
                                --enddate_train 2022-06-01 --startdate_val 2022-07-01
                                --enddate_val 2022-12-01 --startdate_test 2023-01-01
                                --enddate_test 2023-06-01
    """
    
    
    parser = argparse.ArgumentParser(description="Create train, test and validation sets and normalize")

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--tiles", type=str, default="00N_010E,10N_010E",
                        help="Tiles to be processed")
    parser.add_argument("--dump_path", type=str, default="./data_npy_normby_wwf",
                        help="Data dumpth path")
    parser.add_argument('--data_input', type=str, default='C:/PostDoc/deforestation_project/data/tiles_march_2025/input',
                            help="Path containing the input variables")
    parser.add_argument('--data_gt', type=str, default='C:/PostDoc/deforestation_project/data/tiles_march_2025/gt',
                            help="Path containing the groundtruth data")
    parser.add_argument('--dynamic_var',type=str, default=[# ['firealerts', '1000'],
                                                           #['lastsixmonths','1600'],
                                                           #['lastthreemonths','1600'],
                                                           ['lastmonth','1600'],
                                                           #['nightlights','65535'],
                                                           #['patchdensity','800'],
                                                           #['totaldeforestation','1600'],
                                                           ['precipitation','240'],
                                                           ['temperature','3000'],
                                                           #['previoussameseason','1600'],
                                                           #['smoothedsixmonths','1600'],
                                                           #['smoothedtotal','1600'],
                                                           ['timesinceloss','10000'],
                                                           ['totallossalerts','1600']], 
                        help="Names of the dynamic variables to be used")
    parser.add_argument('--auto_gen_dynamic',type=str, default=[['month','12'],
                                                                ['sinmonth','0']],
                                                               # ['monthssince2019', '50']], 
                        help="Variables name for autogenration (dynamic)") 
    parser.add_argument('--yearly_var',type=str, default=[['closenesstoroads', '255'],
                                                          ['losslastyear','256']], 
                        help="Names of the semi-dynamic variables")

    parser.add_argument('--static_names',type=str, default=[['closenesstowaterways','255'],
                                                            ['elevation','8849'],
                                                            ['historicloss','256'],
                                                            ['initialforestcover','10000'],
                                                            ['populationcurrent','20000000'],
                                                            ['populationincrease','20000000'],
                                                            ['slope','4000'],
                                                            ['wetlands','0'],
                                                            ['peatland','15'],
                                                            ['croplandcapacity100p','255'],
                                                            ['croplandcapacitybelow50p','255'],
                                                            ['croplandcapacityover50p','255'],
                                                            ['landpercentage','254'],
                                                            ['forestheight', '50'],
                                                            ['wdpa','0'],
                                                            ['catexcap','1000'],
                                                            ['xy', '0']], 
                        help="Names of the static variables")

    parser.add_argument('--ref_name', type=str, default='groundtruth6m',
                            help="Name of the reference variable")

    parser.add_argument('--mask_name', type=str, default='slope',
                            help="Name of the variable to draw the mask of the region")

    parser.add_argument('--normalize', type=bool, default=True,
                            help="True to normalize by the maximum values given by WWF")

    parser.add_argument("--startdate_train", default='2021-01-01', type=str, 
                        help="End date for training data")
    parser.add_argument("--enddate_train", default='2022-12-01', type=str, 
                        help="End date for training data")
    parser.add_argument("--startdate_val", default='2023-06-01', type=str, 
                        help="Start date for validation samples")
    parser.add_argument("--enddate_val", default='2023-12-01', type=str, 
                        help="End date for validation samples")
    parser.add_argument("--startdate_test", default='2024-01-01', type=str, 
                        help="Start date for test samples")
    parser.add_argument("--enddate_test", default='2024-06-01', type=str, 
                        help="End date for test samples")
#
    main(parser)