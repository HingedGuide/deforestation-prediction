# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os

import numpy as np
import torch

from .logger import create_logger, PD_Stats

import torch.distributed as dist

from osgeo import gdal, osr
import errno
import random
import matplotlib.pyplot as plt


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()


def array2raster(newRasterfn, dataset, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype, options=['COMPRESS=LZW'])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def add_padding_new(img, psize, overl, const = 0):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    if len(img.shape) == 4:
        bands, time, row, col = img.shape
    elif len(img.shape) == 3:
        bands, row, col = img.shape
        time = 0
    if len(img.shape) == 2:
        row, col = img.shape
        row, col = img.shape
        bands = 0
        time = 0
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    # overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    # row += overlap//2
    # col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands>0 and time > 0:
        npad_img = ((0,0),(0,0),(overlap//2, step_row+overlap),(overlap//2, step_col+overlap))
    elif bands>0 and time == 0:       
        npad_img = ((0,0), (overlap//2, step_row+overlap), (overlap//2, step_col+overlap)) 
    elif bands==0 and time == 0:       
        npad_img = ((overlap//2, step_row+overlap), (overlap//2, step_col+overlap)) 
        
    # padd with symetric (espelhado)    
    pad_img = np.pad(img, npad_img, mode='constant', constant_values=const)

    # Number of patches: k1xk2
    k1, k2 = (row+step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap, k1, k2

def extract_patches_coord(gt, psize, ovrl):
    '''Function to extract patches coordinates from rater images
        input:
            img: raster image  
            gt: shpafile raster
            psize: image patch size
            ovrl: overlap to extract patches
            model: model type

    '''
    # add padding to gt raster
    img_gt, stride, step_row, step_col, overlap,_, _ = add_padding_new(gt, psize, ovrl)
    row,col = img_gt.shape
    
    unique_class = np.unique(img_gt[img_gt!=0])
    
    if stride == 1:
        coords = np.where(img_gt!=0)
        coords = np.array(coords)
        coords = np.rollaxis(coords, 1, 0)

    else:
        # loop over x,y coordinates and extract patches
        coord_list = list()
    
        # for m in range(psize//2,row-psize//2,stride): 
        #     for n in range(psize//2,col-psize//2,stride):
        for m in range(psize//2, row - psize//2 + psize % 2, stride):
            for n in range(psize//2, col - psize//2 + psize % 2, stride):
                coord = [m,n]
                class_patch = np.unique(img_gt[m-psize//2:m+psize//2,
                                     n-psize//2:n+psize//2])
                if len(class_patch)>1 or class_patch[0] in unique_class:
                    coord_list.append(coord)                    
                            
        coords = np.array(coord_list)
    
    return coords, img_gt


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cuda:0", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
                
                
def plot_figures(img,mask,pred,model_dir,epoch,set_name):
        img = img.data.cpu().numpy()
        mask = mask.data.cpu().numpy()
        pred = pred.data.cpu().numpy()
        
        batch = 8 if img.shape[0] > 8 else img.shape[0]
        nrows = 3

        
        img = img[:batch,2,:,:]
        pred = pred[:batch,:,:]
        
        mask = mask[:batch,:,:]
        
        if "patch_wise" in model_dir:
            pred = pred[:, np.newaxis, np.newaxis]
            pred = np.pad(pred, ((0,0),(1,1),(1,1)), 'reflect')
            
            mask = mask[:, np.newaxis, np.newaxis]
            mask = np.pad(mask, ((0,0),(1,1),(1,1)), 'reflect')
                            
                            
        fig, axes = plt.subplots(nrows=nrows, ncols=batch, figsize=(batch, nrows))
        
        imgs = [img,mask,pred]
        
        cont = 0
        cont_img = 0
        cont_bacth = 0
        for ax in axes.flat:
            ax.set_axis_off()
            if cont < 1:
                im = ax.imshow(imgs[cont][cont_bacth], cmap='gray', interpolation='nearest')
            elif cont == 1 and "resnet_patch_wise" not in model_dir:
                im = ax.imshow(imgs[cont][cont_bacth], cmap='OrRd', interpolation='nearest', vmin=0, vmax=2)
            else:
                im = ax.imshow(imgs[cont][cont_bacth], cmap='OrRd', interpolation='nearest', vmin=0, vmax=2)
                
            cont_img+=1
            cont_bacth+=1
            if cont_img%batch==0:
                cont+=1
                cont_bacth=0

        
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                            wspace=0.02, hspace=0.02)
        
        plt.axis('off')
        plt.savefig(os.path.join(model_dir, set_name + str(epoch) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
        plt.clf()
        plt.close()


def check_folder(folder_dir):
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_tiff(tiff_file):
    #print(tiff_file)
    data = gdal.Open(tiff_file).ReadAsArray()
    return data


def fun_sort(x):
    return int(x.split('_')[0])


def add_padding(img, psize, val = 0):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        bands, row, col = img.shape
    except:
        bands = 0
        row, col = img.shape
    
    if bands>0:
        npad_img = ((0,0), (psize//2+1, psize//2+1), (psize//2+1, psize//2+1))
        constant_values = val
    else:        
        npad_img = ((psize//2+1, psize//2+1), (psize//2+1, psize//2+1))
        constant_values = val

    pad_img = np.pad(img, npad_img, mode='constant', constant_values=constant_values)

    return pad_img

