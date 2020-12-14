import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from utils import supportingFunctions as sf
import torch
from scipy.io import loadmat
from torch.autograd import Variable
from cascadenet_pytorch.model_pytorch import *
from cascadenet_pytorch.dnn_io import to_tensor_format
from utils import compressed_sensing as cs
import torch.optim as optim
import argparse


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]

def prep_input(im, acc=4):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = torch.from_numpy(to_tensor_format(im))
    im_und_l = torch.from_numpy(to_tensor_format(im_und))
    k_und_l = torch.from_numpy(to_tensor_format(k_und))
    mask_l = torch.from_numpy(to_tensor_format(mask, mask=True))

    return im_und_l, k_und_l, mask_l, im_gnd_l


def create_dummy_data(num):
    data = loadmat(os.path.join('.', './data/cardiac.mat'))['seq']
    nx, ny, nt = data.shape
    ny_red = 8
    sl = ny // ny_red
    data_t = np.transpose(data, (2, 0, 1))
    test = np.expand_dims(data_t[0:num, :, :], 0)
    return test


def get_netoutputs(demo, num, network, presaved):
    
    if demo == 'yes' and not os.path.exists('Demo'+'_outputs/'):
       os.mkdir('Demo' + '_outputs/')

    if not os.path.exists(network+'_outputs/'):
        os.mkdir(network+'_outputs/')

    if network == 'modl':

        cwd = os.getcwd()
        tf.compat.v1.reset_default_graph()

        # %% choose a model from savedModels directory

        subDirectory = '04Jun_0356pm_5L_10K_50E_AG'

        # %%Read the testing data from dataset.hdf5 file

        # tstOrg is the original ground truth
        # tstAtb: it is the aliased/noisy image
        # tstCsm: this is coil sensitivity maps
        # tstMask: it is the undersampling mask

        if demo == 'yes':
            tstOrg, tstAtb, tstCsm, tstMask, y = sf.getTestingData()  # y is the LR image

        else:
            tstOrg, tstAtb, tstCsm, tstMask, y = sf.getData(num, sigma=0)  # y is the measurement

        if demo == 'yes':
            network = 'Demo'
        np.save(os.path.join(network + '_outputs/', network + '_csm.npy'), tstCsm)

        #%% Load existing model. Then do the reconstruction
        print('Now loading the model ...')

        modelDir = cwd+'/models/'+subDirectory #complete path
        rec = np.empty(tstAtb.shape, dtype=np.complex64) #rec variable will have output

        tf.compat.v1.reset_default_graph()
        loadChkPoint=tf.train.latest_checkpoint(modelDir)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.compat.v1.Session(config=config) as sess:
            new_saver = tf.compat.v1.train.import_meta_graph(modelDir+'/modelTst.meta')
            new_saver.restore(sess, loadChkPoint)
            graph = tf.compat.v1.get_default_graph()
            predT = graph.get_tensor_by_name('predTst:0')
            maskT = graph.get_tensor_by_name('mask:0')
            atbT = graph.get_tensor_by_name('atb:0')
            csmT = graph.get_tensor_by_name('csm:0')
            wts=sess.run(tf.compat.v1.global_variables())
            dataDict={atbT:tstAtb,maskT:tstMask,csmT:tstCsm}

            if presaved == 'True':
                rec = np.load('Brain_iter100_10e-5/rec.npy')
            else:
                rec=sess.run(predT,feed_dict=dataDict)
                rec = sf.r2c(rec.squeeze())

        print('Network outputs saved')

    if network== 'crnn':

        batch_size = 1
        lr = 0.001
        acceleration_factor = 4

        cuda = False
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        # Project config
        acc = float(acceleration_factor)  # undersampling rate
        batch_size = int(batch_size)

        test = create_dummy_data(num)

        # Specify network
        rec_net = CRNN_MRI()
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(rec_net.parameters(), lr=float(lr), betas=(0.5, 0.999))

        # build CRNN-MRI with pre-trained parameters
        rec_net.load_state_dict(torch.load('./models/pretrained/crnn_mri_d5_c5.pth', map_location=torch.device('cpu')))

        if cuda:
            rec_net = rec_net.cuda()
            criterion.cuda()

        for im in iterate_minibatch(test, batch_size, shuffle=False):
            im_und, k_und, mask, im_gnd = prep_input(im, acc)

            with torch.no_grad():
                im_u = Variable(im_und.type(Tensor))
                k_u = Variable(k_und.type(Tensor))
                mask = Variable(mask.type(Tensor))
                gnd = Variable(im_gnd.type(Tensor))
                pred = rec_net(im_u, k_u, mask, test=True)
            outputim = pred.detach().numpy()

            k_und = k_und.numpy()
            im_gnd = im_gnd.numpy()
            mask = mask.numpy()

            y = sf.r2c(k_und.squeeze().transpose((3,1,2,0)))
            y = y.reshape((num, -1))
            tstOrg = sf.r2c(im_gnd.squeeze().transpose((3, 1, 2, 0)))
            tstMask = mask.squeeze().transpose((3, 1, 2, 0))[:,:,:,0]
            rec = sf.r2c(outputim .squeeze().transpose((3, 1, 2, 0)))
            # rec = rec.reshape(num, 256, 256)

    np.save(os.path.join(network+'_outputs/', network +'_mask.npy'),tstMask)
    np.save(os.path.join(network+'_outputs/', network + '_rec.npy'), rec)
    np.save(os.path.join(network+'_outputs/', network+'_b.npy'), y)
    np.save(os.path.join(network+'_outputs/', network+'_GT.npy'), tstOrg)






