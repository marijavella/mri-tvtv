
"""
This test code will load an image from the brain dataset stored in the demoImage.hdf5. 
This image is first passed through the pretrained MoDL network and then post-processed 
via TV-TV. To test larger datasets please run test.py.  
"""
import numpy as np
import argparse
import os
from utils import supportingFunctions as sf
from TVTV_Solver import TVTVSolver
from network_outputs import get_netoutputs

parser = argparse.ArgumentParser()
parser.add_argument('--demo', type=str, default='yes', help='yes, no')
parser.add_argument('--beta', type=int, default=1, help='define beta')
parser.add_argument('--network', type=str, default='modl', help='modl, crnn')
parser.add_argument('--rho', type=int, default=0.5, help='define rho for ADMM')
parser.add_argument('--max_iters', type=int, default=100, help='set maximum no. of iterations')
parser.add_argument('--saveimgs', type=str, default='no', help='yes, no')
parser.add_argument('--crop', type=str, default='True', help='True, False')
parser.add_argument('--nsamples', type=int, default=1, help='no. of test images')
parser.add_argument('--presaved', type=str, default='False', help='use presaved outputs?')
parser.add_argument('--showimgs', type=str, default='yes', help='yes, no')

args = parser.parse_args()

# get output from MoDL
get_netoutputs(args.demo, args.nsamples, args.network, args.presaved)

# define paths for the inputs required for the function TVTVSolver
path_mask = os.path.join('Demo_outputs', 'Demo_mask.npy')
path_net = os.path.join('Demo_outputs', 'Demo_rec.npy')
path_b = os.path.join('Demo_outputs', 'Demo_b.npy')
path_GT = os.path.join('Demo_outputs', 'Demo_GT.npy')
path_csm = os.path.join('Demo_outputs', 'Demo_csm.npy')

rec = np.load(path_net)
masks = np.load(path_mask)
csm = np.load(path_csm)
b = np.load(path_b, allow_pickle=True)

M = rec.shape[0] # no. of rows
N = rec.shape[1] # no. of columns
nimgs = 1

print('Starting TV-TV Solver')
tv_modl_all = TVTVSolver(M, N, rec.squeeze(), args.beta, b.squeeze(), masks.squeeze(), args.rho, args.max_iters, csm.squeeze())
tv_modl_all = np.expand_dims(tv_modl_all,0)
rec = np.expand_dims(rec, 0)

print('Computing evaluation metrics')
psnr_tv, ssim_tv, psnr_net, ssim_net = sf.evaluate_metrics(tv_modl_all, rec, nimgs, path_GT, args.saveimgs, args.crop, args.demo, args.network, args.showimgs)

print('********* Mean PSNR/SSIM Results **********')
print('psnr_tv = ', psnr_tv,'dB')
print('ssim_tv = ', ssim_tv)
print('psnr_network = ', psnr_net,'dB')
print('ssim_network = ', ssim_net)
