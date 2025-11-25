import argparse
import math
import os
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import glob
from osgeo import gdal

from src.logger import create_logger

import segmentation_models_pytorch as smp

from src.utils import (
    bool_flag,
    check_folder,
    extract_patches_coord, 
    add_padding_new, 
    array2raster,
    read_tiff
)

from src.models import ResUnet, WeightedFocalLoss, WeightedAsymmetricLoss, WeightedBCELoss
from src.models import WeightedDiceLoss, WeightedJaccardLoss
from src.multicropdataset import DataloaderEval




def main(ov, ar, ls, bal, opt):
    global args, figures_path
    args = parser.parse_args()
    
    args.arch = ar
    args.loss_fun = ls
    args.balance = bal
    args.optimizer = opt
    
    country = args.dump_path.split('/')[-1]
    
    try:
        args.tiles = args.tiles.split(",")  # Convert the comma-separated string into a list
    except:
        dummpy_var = 0
    
    path_gt_save = args.dump_path
    args.dump_path = os.path.join(args.dump_path, args.arch, '{}_{}_{}_{}'.format(args.size_crops,args.loss_fun,args.balance,args.optimizer))
    
    args.pretrained = os.path.join(args.dump_path,args.pretrained)
    
    args.overlap = ov

    ######## get data #####

    
    # create a logger
    logger = create_logger(os.path.join(args.dump_path, "inference.log"),rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )


    # build model
    if 'resunet' in args.arch:
        model = ResUnet(channel = args.features,
                        classes=1, 
                        filters=args.filters)
        
    elif 'deeplabv3+' in args.arch:
        model = smp.DeepLabV3Plus(
            encoder_name='resnet18', 
            encoder_weights="imagenet", 
            classes=1, 
            activation=None,
        )
        model.encoder.conv1 = nn.Conv2d(args.features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.segmentation_head[1] = nn.UpsamplingBilinear2d(scale_factor=2.0)
        
    elif 'deeplabv3resnet50' in args.arch:          
        model = torch.hub.load('pytorch/vision:v0.10.0', "deeplabv3_resnet50", 
                                pretrained=True,
                                aux_loss =False)
        model.backbone.conv1 = nn.Conv2d(args.features, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        
    elif 'resnet' in args.arch:
        arch = 'resnet18'
        model = torch.hub.load('pytorch/vision:v0.10.0', arch, weights='ResNet18_Weights.DEFAULT')
        model.conv1 = nn.Conv2d(args.features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        
    # model to gpu
    model = model.cuda()
    
    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:0", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    cudnn.benchmark = True
    
    model.eval()

    check_folder(os.path.join(args.dump_path,'prediction'))
    
    for tile in args.tiles:
        for setp in args.set_2pred:
            test_data = np.load(os.path.join(args.image_path, f'{tile}_var_{setp}.npy'))
            if setp == 'train':
                gt_data = np.load(os.path.join(args.image_path, f'{tile}_gt_tr.npy'))
                test_data = test_data[:,-12:,:,:]
                gt_data = gt_data[-12:,:,:]
                
            else:
                gt_data = np.load(os.path.join(args.image_path, f'{tile}_gt_{setp}.npy'))
            
            mask = np.load(os.path.join(args.image_path, f'{tile}_mask.npy')).astype('float16')
            
            mask_country = (os.path.join(args.image_path, f'{tile}_mask_{country}.tiff'))
            mask_country = read_tiff(mask_country)
            
            test_data[np.isnan(test_data)] = 0
            
            test_data, gt_data, coords, stride, step_row, step_col, overlap = define_loader(test_data, gt_data, mask_country)
            
            bands, time, row, col = test_data.shape
            
    
            pred_prob = np.zeros(shape = (row, col, time), dtype='float16')
            ref_data = np.zeros(shape = (row, col, time), dtype='uint8')
            
            
            
            for t in range(time):
        
                # define loader
                val_dataset = DataloaderEval(test_data[:,t,:,:],
                                          coords,
                                          psize = args.size_crops,
                                          labels = gt_data[t,:,:], 
                                          lab_ind=True)
                
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False
                )
                    
                logger.info("Building data done for time-step {}".format(t))
                
                pred_prob, ref_data = predict_network(val_loader, model, coords, pred_prob, ref_data, stride, step_row, step_col, overlap, t, logger)
                
            raster_src = gdal.Open(os.path.join(args.ref_raster, f'{tile}/{tile}_2004-01-01_wetlands.tif'))
            
            row, col = raster_src.RasterYSize, raster_src.RasterXSize
            
            pred_prob = pred_prob[overlap//2+1:,overlap//2+1:]
            pred_prob = pred_prob[:mask.shape[0],:mask.shape[1]]  
            
            ref_data = ref_data[overlap//2+1:,overlap//2+1:]
            ref_data = ref_data[:mask.shape[0],:mask.shape[1]]
            
    #        np.save(os.path.join(args.dump_path,'prediction',f'prob_map_{tile}_{args.set_2pred}'), pred_prob.astype('float16'))
            array2raster(os.path.join(args.dump_path,'prediction',f'prob_map_{tile}_{setp}.tiff'.format(args.overlap)), raster_src, pred_prob, "Float32")
            array2raster(os.path.join(path_gt_save,f'ref_map_{tile}_{setp}.tiff'), raster_src, ref_data, "Byte")
            
    
    logger.info("============ Inference finished ============")

def predict_network(dataloader, model, coords, pred_prob, ref_data,
                     stride, step_row, step_col, overlap, timestep, logger):
      
    model.eval()
    
    sig = nn.Sigmoid()
    
    st = stride//2
    ovr = overlap//2
    
    j = 0
    with torch.no_grad(): 
        for i, inputs in enumerate(dataloader):      
            # ============ multi-res forward passes ... ============
            # compute model loss and output
            input_batch = inputs[0].cuda(non_blocking=True)
            out_batch = model(input_batch).cuda(non_blocking=True)
               
            out_batch = sig(out_batch)

            out_batch = out_batch[:,0,:,:].data.cpu().numpy()
            ref = inputs[1].data.cpu().numpy()
            
            
            c, x, y = out_batch.shape
            coord_x = coords[j:j+args.batch_size,0]
            coord_y = coords[j:j+args.batch_size,1]
            for b in range(c):
                pred_prob[int(coord_x[b] - st): int(coord_x[b] + st + stride % 2),
                                        int(coord_y[b] - st): int(coord_y[b] + st + stride % 2),timestep] = \
                                        out_batch[b, int(overlap // 2): int(x - ovr - overlap % 2),
                                                  int(overlap // 2):int(y - ovr - overlap % 2)]
                ref_data[int(coord_x[b] - st): int(coord_x[b] + st + stride % 2),
                                        int(coord_y[b] - st): int(coord_y[b] + st + stride % 2),timestep] = \
                                        ref[b, int(overlap // 2): int(x - ovr - overlap % 2),
                                                  int(overlap // 2):int(y - ovr - overlap % 2)]
            
            j+=out_batch.shape[0] 
            
        
        return pred_prob, ref_data

        
def define_loader(image, gt_data, mask):
        
    coords, _ = extract_patches_coord(mask, args.size_crops, args.overlap)
    gt_data, _, _, _, _, _, _ = add_padding_new(gt_data, args.size_crops, args.overlap)
    image, stride, step_row, step_col, overlap, _, _ = add_padding_new(image, args.size_crops, args.overlap)

    return image, gt_data, coords, stride, step_row, step_col, overlap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference of FCN")
    
    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--dump_path", type=str, default="./exp_tiles_continent_server/Africa",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument('--image_path', type=str, default='./data_npy_normby_wwf',
                            help="Path containing the numpy array with the input image")
    parser.add_argument('--ref_path',type=str, default='./data_npy_normby_wwf',
                            help="Path containing the numpy array with the input image")
    parser.add_argument('--set_2pred',type=str, default=['val','test'],
                            help="Set to predict: test or val")
    parser.add_argument("--overlap", type=float, default=[0.2,0.4,0.6], 
                        help="samples per epoch")
    parser.add_argument("--size_crops", type=int, default=64, 
                        help="Crop size")
    parser.add_argument("--features", type=int, default=35, 
                        help="Number of features")
    parser.add_argument('--ref_raster', type=str, default='C:/PostDoc/deforestation_project/data/tiles_march_2025/input',
                            help="Path containing the numpy array with the input image")
    
    parser.add_argument('--tiles',type=str, default="10N_000E,10N_010E,10N_020E,00N_000E,00N_010E,00N_020E,10N_030E", 
                        help="Path containing the tile name to be loaded")
    
    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--arch", default="resunet", type=str, 
                        help="convnet architecture --> 'resunet','deeplabv3+','resnet'")
    parser.add_argument("--filters", default=[16,32,64,128], type=int, 
                        help="Filter for the ResUnet for trained from scratch")
    parser.add_argument("--patch_wise", type=bool, default=False, 
                        help="Set True for CNN-patch")
    
    ##########################
    #### others parameters ###
    ##########################
    parser.add_argument("--workers", default=0, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=1,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=bool_flag, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    
    #########################
    #### model parameters ###
    #########################
    parser.add_argument("--pretrained", default="checkpoint.pth.tar", type=str, 
                        help="path to pretrained weights")
    
    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size ")

    
    overlaps = [0.2]
    arch = ['resunet']
    loss_fun = ["dice+cross"]
    # loss_fun = ['dice',"focal+dice","focal+cross","dice+cross"]
    balance = [1]  # not balance at all, 1 balance coordinates, 3 balance coords + add weight
    optimizer = ['RAdam']

    for ov in overlaps:
        for ar in arch:
            for ls in loss_fun:
                for bal in balance:
                    for opt in optimizer:
                        main(ov, ar, ls, bal, opt)
