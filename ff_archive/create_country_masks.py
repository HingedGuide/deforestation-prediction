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
from src.logger import create_logger

import numpy as np
import glob
from pathlib import Path
from datetime import datetime

import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask

from src.utils import (
    read_tiff,
    check_folder,
)


def main(parser):
    global args
    args = parser.parse_args()
    check_folder(args.dump_path)
    
    args.tiles = args.tiles.split(",")  # Convert the comma-separated string into a list
    
    logger = create_logger(os.path.join(args.dump_path, "creating_masks.log"),rank=0)
    logger.info("============ Initialized logger ============")
    
    for tile in args.tiles:
        
        if os.path.isfile(os.path.join(args.dump_path,'{}_mask_{}.tiff'.format(tile, args.country[1]))):
             logger.info(f"Mask for {tile} country {args.country} already processed")
             continue
        else:
            
            # Read the shapefile
            gdf = gpd.read_file(args.shapefile)
            
            # Select a polygon by name (adjust column name as needed)
            if args.column == 'all':
                # Select all polygons
                selected_polygon = gdf  # Selecting all polygons
            else:
                polygon_name = args.country  # Replace with the actual polygon name
                selected_polygon = gdf[gdf[args.column] == polygon_name]  # Replace 'name_column' with the actual column name
            
            if selected_polygon.empty:
                raise ValueError(f"Polygon with name '{polygon_name}' not found in shapefile.")
                
            tiff_path = glob.glob(os.path.join(args.data_input, tile) + '/*_{}'.format(args.mask_name) + '*.tif')[0]
            
            output_mask_path = os.path.join(args.dump_path,f'{tile}_mask_{args.country}.tiff')
            
            # Read the reference raster
            with rasterio.open(tiff_path) as src:
                transform = src.transform
                out_shape = (src.height, src.width)
                crs = src.crs
            
                # Reproject the selected polygon if needed
                selected_polygon = selected_polygon.to_crs(crs)
                
                # Create a binary mask
                mask = geometry_mask(geometries=selected_polygon.geometry, transform=transform, invert=True, out_shape=out_shape)
            
                # Save the binary mask
                with rasterio.open(
                    output_mask_path, 'w',
                    driver='GTiff',
                    height=out_shape[0],
                    width=out_shape[1],
                    count=1,
                    dtype=rasterio.uint8,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(mask.astype(rasterio.uint8), 1)
    


if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description="Create train, test and validation sets and normalize")

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--tiles", type=str, default="10N_090E,10N_100E,00N_100E,10N_110E,00N_110E,10N_120E,00N_120E,00N_130E,00N_140E,00N_090E,10S_140E,10S_150E,00N_150E",
                        help="Tiles to be processed")
    parser.add_argument("--dump_path", type=str, default="./data_npy_normby_wwf",
                        help="Data dumpth path")
    parser.add_argument('--data_input', type=str, default='C:/PostDoc/deforestation_project/data/tiles_march_2025/input',
                            help="Path containing the input variables")
    parser.add_argument('--shapefile', type=str, default='C:/PostDoc/deforestation_project/Laura_CNN_experiments/asia_shape.shp',
                            help="Shapefile to load to get the country")
    parser.add_argument('--country', type=str, default='Asia',
                            help="Column name for the shapefile")
    parser.add_argument('--column', type=str, default='all',
                            help="Column name")
    parser.add_argument('--mask_name', type=str, default='slope',
                            help="Name of the variable to draw the mask of the region")

    main(parser)