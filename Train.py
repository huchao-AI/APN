import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.base_model import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore") 

from model.ssim_loss import SSIM


parser = argparse.ArgumentParser(description="MPN")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=3000, help='number of epochs for training')
parser.add_argument('--loss_fra_reconstruct', type=float, default=1.00, help='weight of the frame reconstruction loss')
parser.add_argument('--loss_fea_reconstruct', type=float, default=1.00, help='weight of the feature reconstruction loss')
parser.add_argument('--loss_distinguish', type=float, default=0.0001, help='weight of the feature distinction loss')
parser.add_argument('--loss_sim', type=float, default=0.0001, help='weight of the feature sim loss')
parser.add_argument('--loss_ssim', type=float, default=0.0001, help='weight of the img ssim loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr_D', type=float, default=1e-3, help='initial learning rate for parameters')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--segs', type=int, default=32, help='num of video segments')
parser.add_argument('--fdim', type=list, default=[128], help='channel dimension of the features')
parser.add_argument('--pdim', type=list, default=[128], help='channel dimension of the prototypes')
parser.add_argument('--psize', type=int, default=10, help='number of the prototype items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='/data/VAD/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--resume', type=str, default='exp/ped2/example.pth', help='file path of resume pth')
parser.add_argument('--prtrain', type=str, default='exp/ped2/example.pth', help='file path of prtrain pth')
parser.add_argument('--debug', type=bool, default=False, help='if debug')
args = parser.parse_args()

torch.manual_seed(2021)
print(args)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+args.dataset_type+"/training/frames"

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)


train_size = len(train_dataset)
print(train_size)
train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)


# Model setting
model = convAE(args.c, args.t_length, args.psize, args.fdim[0], args.pdim[0])
model.cuda()


params_extract = list(model.extract.parameters())
params_proto = list(model.prototype.parameters())
params_output = list(model.ohead.parameters())
params_D =  params_extract + params_output + params_proto
#params_D =  params_output + params_proto

optimizer_D = torch.optim.AdamW(params_D, lr=args.lr_D)


start_epoch = 0
if os.path.exists(args.resume):
  print('Resume model from '+ args.resume)
  ckpt = args.resume
  checkpoint = torch.load(ckpt)
  start_epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['state_dict'].state_dict())
  optimizer_D.load_state_dict(checkpoint['optimizer_D'])

if os.path.exists(args.prtrain):
  print('prtrain model from '+ args.prtrain)
  ckpt = args.prtrain
  checkpoint = torch.load(ckpt)
  model.load_state_dict(checkpoint['state_dict'].state_dict())

if len(args.gpus[0])>1:
  model = nn.DataParallel(model)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not args.debug:
  orig_stdout = sys.stdout
  f = open(os.path.join(log_dir, 'log.txt'),'w')
  sys.stdout= f

loss_func_mse = nn.MSELoss(reduction='none')
loss_func_ssim = SSIM()
loss_pix = AverageMeter()
loss_fea = AverageMeter()
loss_dis = AverageMeter()
loss_sim = AverageMeter()
loss_ssim = AverageMeter()
# Training

model.train()

for epoch in range(start_epoch, args.epochs):
    labels_list = []
    
    pbar = tqdm(total=len(train_batch))
    for j,(imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()
        imgs = imgs.view(args.batch_size,-1,imgs.shape[-2],imgs.shape[-1])

        outputs, _, _, _, fea_loss, _, dis_loss, sim_loss = model.forward(imgs[:,0:12], None, True)
        optimizer_D.zero_grad()
        
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
        fea_loss = fea_loss.mean()
        dis_loss = dis_loss.mean()

        ssim_loss = 1- torch.mean(loss_func_ssim(outputs, imgs[:,12:]))

        loss_D = args.loss_fra_reconstruct*loss_pixel + args.loss_ssim*ssim_loss + args.loss_fea_reconstruct * fea_loss + args.loss_distinguish * dis_loss + args.loss_sim * sim_loss
        loss_D.backward(retain_graph=True)
        optimizer_D.step()


        loss_pix.update(args.loss_fra_reconstruct*loss_pixel.item(), 1)
        loss_fea.update(args.loss_fea_reconstruct*fea_loss.item(), 1)
        loss_dis.update(args.loss_distinguish*dis_loss.item(), 1)
        loss_sim.update(args.loss_sim * sim_loss.item(), 1)
        loss_ssim.update(args.loss_ssim * ssim_loss.item(), 1)

        pbar.set_postfix({
                      'Epoch': '{0} {1}'.format(epoch+1, args.exp_dir),
                      'Lr': '{:.6f}'.format(optimizer_D.param_groups[-1]['lr']),
                      'PRe': '{:.6f}({:.4f})'.format(loss_pixel.item(), loss_pix.avg),
                      'FRe': '{:.6f}({:.4f})'.format(fea_loss.item(), loss_fea.avg),
                      'Dist': '{:.6f}({:.4f})'.format(dis_loss.item(), loss_dis.avg),
                      'sim': '{:.6f}({:.4f})'.format(sim_loss.item(), loss_sim.avg),
                      'ssim': '{:.6f}({:.4f})'.format(ssim_loss.item(), loss_ssim.avg),
                    })
        pbar.update(1)

    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Lr: {:.6f}'.format(optimizer_D.param_groups[-1]['lr']))
    print('PRe: {:.6f}({:.4f})'.format(loss_pixel.item(), loss_pix.avg))
    print('FRe: {:.6f}({:.4f})'.format(fea_loss.item(), loss_fea.avg))
    print('Dist: {:.6f}({:.4f})'.format(dis_loss.item(), loss_dis.avg))
    print('Sim: {:.6f}({:.4f})'.format(sim_loss.item(), loss_sim.avg))
    print('SSim: {:.6f}({:.4f})'.format(ssim_loss.item(), loss_ssim.avg))
    print('----------------------------------------')   

    pbar.close()

    loss_pix.reset()
    loss_fea.reset()
    loss_dis.reset()
    loss_sim.reset()
    loss_ssim.reset()

    # Save the model
    if (epoch + 1)%30==0:
      
      if len(args.gpus[0])>1:
        model_save = model.module
      else:
        model_save = model
        
      state = {
            'epoch': epoch,
            'state_dict': model_save,
            'optimizer_D' : optimizer_D.state_dict(),
          }
      torch.save(state, os.path.join(log_dir, 'model_'+str(epoch + 1)+'.pth'))

    
print('Training is finished')
if not args.debug:
  sys.stdout = orig_stdout
  f.close()
