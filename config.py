import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = 'VISDRONE'  # dataset selection: NWPU, SHHA, SHHB, QNRF, FDST


__C.NET = 'HR_Net' #  optional ['HR_Net', 'VGG16_FPN']

__C.PRE_HR_WEIGHTS = './weights/hrnetv2_w48_imagenet_pretrained.pth'
#__C.PRE_HR_WEIGHTS = './exp/04-15_20-08_VISDRONE_HR_Net/ep_57_F1_0.630_Pre_0.723_Rec_0.558_mae_38.2_mse_50.0.pth'
#__C.PRE_HR_WEIGHTS = './exp/04-12_07-37_VISDRONE_HR_Net/ep_25_F1_0.000_Pre_0.000_Rec_0.000_mae_167.4_mse_183.2.pth'

__C.RESUME = False  # contine training
__C.RESUME_PATH = './exp/04-15_20-08_VISDRONE_HR_Net/latest_state.pth'

__C.GPU_ID = [0]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]
#__C.GPU_ID = 'CPU'

__C.OPT = 'Adam'  #'Adam'
# learning rate settings
if __C.OPT == 'Adam':
    __C.LR_BASE_NET = 1e-5  # learning rate
    __C.LR_BM_NET = 1e-6  # learning rate
__C.LR_DECAY = 0.99 # no use 
__C.NUM_EPOCH_LR_DECAY = 4 # no use 
__C.LR_DECAY_START = 10 # no use 

__C.MAX_EPOCH = 200
__C.PRINT_FREQ = 20

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET 


__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes

#------------------------------VAL------------------------
__C.VAL_DENSE_START = 0
__C.VAL_FREQ = 2  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes


#================================================================================
#================================================================================
#================================================================================
