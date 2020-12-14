"""
This is the code used to reproduce the results obtained in:

Overcoming Measurement Inconsistency in Deep Learning for Linear Inverse Problems: Applications in Medical Imaging
Marija Vella, João F. C. Mota
https://arxiv.org/abs/2011.14387

Testing on MoDL:

Keep the defualt settings to pass the images through the pretrained MoDL network and afterwards post process them
via TV-TV minimization. If you would like to use the results presented in the paper for MoDL set --presaved == True. 

Testing on CRNN:

Use the following settings:
python test.py --network crnn --beta 0.8 --nsamples 30 --crop False --multi_coil False
"""
import os
import numpy as np
import argparse
import multiprocessing
from utils import supportingFunctions as sf
from TVTV_Solver import TVTVSolver
from joblib import delayed, Parallel
from network_outputs import get_netoutputs

# Processing images in parallel if multiple CPU cores are available
def get_tvtvout(M, N, rec, beta, y, tstMask, nimgs, rho, max_iter, tstCsm):
    
    num_cores = multiprocessing.cpu_count()

    if tstCsm is None:
        tvtv_out = Parallel(n_jobs=num_cores)(delayed(TVTVSolver)(M, N, rec[i,:,:], beta, y[i],
                    np.squeeze(tstMask[i,:,:]), rho, max_iter) for i in range(nimgs))
    else:
        tvtv_out = Parallel(n_jobs=num_cores)(delayed(TVTVSolver)(M, N, rec[i,:,:], beta, y[i],
                    np.squeeze(tstMask[i,:,:]), rho, max_iter, np.squeeze(tstCsm[i,:,:,:])) for i in range(nimgs))
    return tvtv_out

parser = argparse.ArgumentParser()
parser.add_argument('--demo', type=str, default='no', help='yes, no')
parser.add_argument('--multi_coil', type=str, default='True', help='True, False')
parser.add_argument('--crop', type=str, default='True', help='True, False')
parser.add_argument('--network', type=str, default='modl', help='modl, crnn')
parser.add_argument('--beta', type=float, default=1, help='define beta')
parser.add_argument('--nsamples', type=int, default=163, help='no. of test images')
parser.add_argument('--rho', type=float, default=0.4, help='define rho for ADMM')
parser.add_argument('--max_iters', type=int, default=100, help='set maximum no. of iterations')
parser.add_argument('--saveimgs', type=str, default='no', help='yes, no')
parser.add_argument('--showimgs', type=str, default='no', help='yes, no')
parser.add_argument('--path_results', type=str, default='Results', help='path for results')
parser.add_argument('--presaved', type=str, default='False', help='use presaved outputs?')

args = parser.parse_args()

get_netoutputs(args.demo, args.nsamples, args.network, args.presaved)

path_mask = os.path.join(args.network+'_outputs/', args.network + '_mask.npy')
path_net = os.path.join(args.network+'_outputs/', args.network + '_rec.npy')
path_b = os.path.join(args.network+'_outputs/', args.network+'_b.npy')
path_GT = os.path.join(args.network+'_outputs/', args.network+'_GT.npy')

if args.multi_coil == 'True':
    path_csm = os.path.join(args.network+'_outputs/', args.network+'_csm.npy')
    csm = np.load(path_csm)
else:
    csm = None

rec = np.load(path_net)
masks = np.load(path_mask)
b = np.load(path_b,  allow_pickle=True)

nimgs = rec.shape[0] # no. of images
M = rec.shape[1] # no. of rows
N = rec.shape[2] # no. of columns

if args.presaved == 'True':
    # load the generated TV-TV outputs as reported in the paper.  
    tv_modl_all = np.load('Brain_iter100_10e-5/solver_out.npy')

else:
    print('Starting TV-TV Solver')
    tv_modl_all = get_tvtvout(M, N, rec, args.beta, b, masks, nimgs, args.rho, args.max_iters, csm)

psnr_tv, ssim_tv, psnr_net, ssim_net = sf.evaluate_metrics(tv_modl_all, rec, nimgs, path_GT, args.saveimgs,
                                                           args.crop, args.demo, args.network, args.showimgs)

print('********* Mean PSNR/SSIM Results **********')
print('psnr_tv = ', psnr_tv,'dB')
print('ssim_tv = ', ssim_tv)
print('psnr_network = ', psnr_net,'dB')
print('ssim_network = ', ssim_net)
