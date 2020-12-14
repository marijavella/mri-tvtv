# Modified from https://github.com/hkaggarwal/modl/blob/master/supportingFunctions.py

import numpy as np
import h5py as h5
import tensorflow as tf
from utils import supportingFunctions as sf
from skimage.metrics import structural_similarity as ssim
import os.path
import matplotlib.pyplot as plt
from utils import*


def evaluate_metrics(tv_modl_all, rec, nimgs, path_gt, save_imgs, crop, demo, network, show_imgs):

    tstOrg = np.load(path_gt)

    all_tv_psnr = []
    all_tv_ssim = []
    all_modl_psnr = []
    all_modl_ssim = []

    fn = lambda x: sf.normalize01(np.abs(x))
    normtv_modl = fn(np.asarray(tv_modl_all))
    normOrg = fn(tstOrg)
    normRec = fn(rec)

    # plt.imshow(np.abs(normtv_modl[0]))
    # plt.title('TVTV-0')
    # plt.show()
    if len(normtv_modl.shape) < 3:
        normtv_modl = np.expand_dims(normtv_modl, 0)
        normOrg = np.expand_dims(normOrg, 0)
        normRec = np.expand_dims(normRec, 0)

    normtv_modl = normtv_modl.transpose((1, 2, 0))
    normOrg = normOrg.transpose((1, 2, 0))
    normRec = normRec.transpose((1, 2, 0))

    for i in range(nimgs):

        if crop == 'True':
            s1 = list(np.sort(np.where((normOrg[:, :, i] > 0.25))[0]))
            s2 = list(np.sort(np.where((normOrg[:, :, i] > 0.25))[1]))
            y1, y2 = list(dict.fromkeys(s1))[0] - 10, list(dict.fromkeys(s1))[-1] + 10
            x1, x2 = list(dict.fromkeys(s2))[0] - 10, list(dict.fromkeys(s2))[-1] + 10

            if x1 < 0:
               x1 = 0
        else:
            y1, y2 = 0, normOrg.shape[0]
            x1, x2 = 0, normOrg.shape[1]

        psnrRec_tvmodl = sf.myPSNR(normOrg[y1:y2, x1:x2, i], normtv_modl[y1:y2, x1:x2, i])
        all_tv_psnr.append(psnrRec_tvmodl)
        ssim_tv_modl = ssim(normOrg[y1:y2, x1:x2, i], normtv_modl[y1:y2, x1:x2, i],
                       gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
        all_tv_ssim.append(ssim_tv_modl)

        psnr_modl = sf.myPSNR(normOrg[y1:y2, x1:x2, i], normRec[y1:y2, x1:x2, i])
        all_modl_psnr.append(psnr_modl)
        ssim_modl = ssim(normOrg[y1:y2, x1:x2, i], normRec[y1:y2, x1:x2, i],
                            gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
        all_modl_ssim.append(ssim_modl)

    all_ssim_tv = np.asarray(all_tv_ssim)
    all_psnrRec_tvmodl = np.asarray(all_tv_psnr)
    psnrRec_modl = np.asarray(all_modl_psnr)
    ssim_modl = np.asarray(all_modl_ssim)

    if demo =='yes':
        path_results = 'Demo_results'
    else:
        path_results = os.path.join(network + '_results')

    if not os.path.exists(path_results):
        os.mkdir(path_results)

    np.save(os.path.join(path_results, 'psnr_tv.npy'), all_psnrRec_tvmodl)
    np.save(os.path.join(path_results,'ssim_tv.npy'), all_ssim_tv)
    np.save(os.path.join(path_results, network + '_psnr.npy'), psnrRec_modl)
    np.save(os.path.join(path_results, network + '_ssim.npy'), ssim_modl)

    if save_imgs == 'yes' or show_imgs == 'yes':
        if not os.path.exists(network + '_images/'):
            os.mkdir(network + '_images/')

        for i in range(nimgs):
            plot = lambda x: plt.imshow(x, cmap='gray')
            plt.clf()
            plt.subplot(131)
            plot((normOrg[:, :, i]))
            plt.axis('off')
            plt.title('GT')
            plt.subplot(132)
            plot(normtv_modl[:, :, i])
            plt.title('TV,' + str(all_psnrRec_tvmodl[i].round(2)) + 'dB')
            plt.axis('off')
            plt.subplot(133)
            plot(normRec[:, :, i])
            plt.title('CNN,' + str(psnrRec_modl[i].round(2)) + 'dB')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
            if save_imgs == 'yes':
                plt.show()
            plt.savefig(os.path.join(network + '_images/', 'im_' + str(i)))
            if show_imgs == 'yes':
                plt.show()

    return np.mean(all_psnrRec_tvmodl), np.mean(all_ssim_tv), np.mean(psnrRec_modl), np.mean(ssim_modl)

def mse(x, y):
    return np.mean(np.abs(x - y)**2)

#%%
def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c

#%%
def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i]=div0(img[i]-img[i].min(),img[i].ptp())
        #img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)


#%%
def myPSNR(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed  images"""
    mse=np.sum(np.square(np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)))
    return psnr


#%% Here I am reading the dataset for training and testing from dataset.hdf5 file
def getData(num, sigma=0):
    #num: set this value between 0 to 163. There are total testing 164 slices in testing data
    print('Reading the data. Please wait...')
    filename='data/dataset.hdf5' #set the correct path here

    with h5.File(filename) as f:
        org,csm,mask=f['tstOrg'][0:num],f['tstCsm'][0:num],f['tstMask'][0:num]
        na=np.newaxis
        org,csm,mask=org[na],csm[na],mask[na]

    print('Successfully read the data from file!')
    org = np.squeeze(org)
    csm = np.squeeze(csm)
    mask = np.squeeze(mask)

    atb, y = generateUndersampled(org,csm,mask,sigma)

    print('Successfully undersampled data!')
    atb = c2r(atb)

    return org,atb,csm, mask, y

#Here I am reading one single image from  demoImage.hdf5 for testing demo code
def getTestingData():
    print('Reading the data. Please wait...')
    filename='data/demoImage.hdf5' #set the correct path here

    with h5.File(filename,'r') as f:
        org,csm,mask=f['tstOrg'][:],f['tstCsm'][:],f['tstMask'][:]

    print('Successfully read the data from file!')
    print('Now doing undersampling....')
    atb, y = generateUndersampled(org,csm,mask,sigma=0)
    atb = c2r(atb)
    print('Successfully undersampled data!')
    return org,atb,csm,mask, y

#%%
def piA(x,csm,mask,nrow,ncol,ncoil):
    """ This is a the A operator as defined in the paper"""
    ccImg = (np.reshape(x,(nrow,ncol)))
    coilImages = (np.tile(ccImg,[ncoil,1,1])*csm)
    kspace = (np.fft.fft2((coilImages))/np.sqrt(nrow * ncol))
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))
    res=kspace[mask!=0]
    return res


def piAt(kspaceUnder,csm,mask,nrow,ncol,ncoil):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))
    temp[mask!=0]=kspaceUnder
    img = (np.fft.ifft2(temp)*np.sqrt(nrow*ncol))
    coilComb=np.sum(img*np.conj(csm),axis=0).astype(np.complex64)
    # coilComb=coilComb.ravel();
    return coilComb

def A_single(x, mask, M, N, norm='ortho'):
    x = x.reshape((M, N))
    x_f = mymath.fft2(x, norm=norm)
    x_fu = mask * x_f
    x_fu = x_fu.flatten()
    return x_fu

def At_single(x, mask, M, N, norm='ortho'):
    x = x.reshape((M, N))
    x_u = mask * x
    x_u = mymath.ifft2(x_u, norm=norm)
    x_u = x_u.flatten()
    return x_u

def generateUndersampled(org,csm,mask,sigma=0):

    nSlice, ncoil, nrow, ncol = csm.shape
    atb = np.empty(org.shape,dtype=np.complex64)
    lr = []
    for i in range(nSlice):
        A = lambda z: piA(z,csm[i],mask[i],nrow,ncol,ncoil)
        At = lambda z: piAt(z,csm[i],mask[i],nrow,ncol,ncoil)

        sidx=np.where(mask[i].ravel()!=0)[0]
        nSIDX=len(sidx)
        noise=np.random.randn(nSIDX*ncoil,)+1j*np.random.randn(nSIDX*ncoil,)
        noise = noise*(sigma/np.sqrt(2.))
        #y = A(org[i]) + noise
        y = A(org[i])
        atb[i] = At(y)
        lr.append(y)
    return atb, lr


#%%
def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros( inp.shape[0:2],dtype=dtype)
    out=inp[...,0]+1j*inp[...,1]
    return out

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    return out

#%%
def getWeights(wtsDir,chkPointNum='last'):
    """
    Input:
        wtsDir: Full path of directory containing modelTst.meta
        nLay: no. of convolution+BN+ReLu blocks in the model
    output:
        wt: numpy dictionary containing the weights. The keys names ae full
        names of corersponding tensors in the model.
    """
    tf.reset_default_graph()
    if chkPointNum=='last':
        loadChkPoint=tf.train.latest_checkpoint(wtsDir)
    else:
        loadChkPoint=wtsDir+'/model'+chkPointNum
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as s1:
        saver = tf.train.import_meta_graph(wtsDir + '/modelTst.meta')
        saver.restore(s1, loadChkPoint)
        keys=[n.name+':0' for n in tf.get_default_graph().as_graph_def().node if "Variable" in n.op]
        var=tf.global_variables()

        wt={}
        for key in keys:
            va=[v for v in var if v.name==key][0]
            wt[key]=s1.run(va)

    tf.reset_default_graph()
    return wt

def mySSIM(org,rec):
    """
    org and rec are 3D arrays in range [0,1]
    """
    shp=org.shape
    if np.ndim(org)>=3:
        nimg=np.prod(shp[0:-2])
    elif np.ndim(org)==2:
        nimg=1
    org=np.reshape(org,(nimg,shp[-2],shp[-1]))
    rec=np.reshape(rec,(nimg,shp[-2],shp[-1]))

    ssim=np.empty((nimg,),dtype=np.float32)
    for i in range(nimg):
        ssim[i]=ssim(org[i],rec[i],data_range=org[i].max())
    return ssim

def assignWts(sess1,nLay,wts):
    """
    Input:
        sess1: it is the current session in which to restore weights
        nLay: no. of convolution+BN+ReLu blocks in the model
        wts: numpy dictionary containing the weights
    """

    var=tf.global_variables()
    #check lam and beta; these for for alternate strategy scalars

    #check lamda 1
    tfV=[v for v in var if 'lam1' in v.name and 'Adam' not in v.name]
    npV=[v for v in wts.keys() if 'lam1' in v]
    if len(tfV)!=0 and len(npV)!=0:
        sess1.run(tfV[0].assign(wts[npV[0]] ))
    #check lamda 2
    tfV=[v for v in var if 'lam2' in v.name and 'Adam' not in v.name]
    npV=[v for v in wts.keys() if 'lam2' in v]
    if len(tfV)!=0 and len(npV)!=0:  #in single channel there is no lam2 so length is zero
        sess1.run(tfV[0].assign(wts[npV[0]]))

    # assign W,b,beta gamma ,mean,variance
    #for each layer at a time
    for i in np.arange(1,nLay+1):
        tfV=[v for v in var if 'conv'+str(i) +str('/') in v.name \
             or 'Layer'+str(i)+str('/') in v.name and 'Adam' not in v.name]
        npV=[v for v in wts.keys() if  ('Layer'+str(i))+str('/') in v or'conv'+str(i)+str('/') in v]
        tfv2=[v for v in tfV if 'W:0' in v.name]
        npv2=[v for v in npV if 'W:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'b:0' in v.name]
        npv2=[v for v in npV if 'b:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'beta:0' in v.name]
        npv2=[v for v in npV if 'beta:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'gamma:0' in v.name]
        npv2=[v for v in npV if 'gamma:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'moving_mean:0' in v.name]
        npv2=[v for v in npV if 'moving_mean:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
        tfv2=[v for v in tfV if 'moving_variance:0' in v.name]
        npv2=[v for v in npV if 'moving_variance:0' in v]
        if len(tfv2)!=0 and len(npv2)!=0:
            sess1.run(tfv2[0].assign(wts[npv2[0]]))
    return sess1

