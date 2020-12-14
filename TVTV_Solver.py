####################TV-TV Solver for Complex Vectors######################
# Solves
#                  minimize    TV(x) + beta*TV(x - w_im)
#                      x
#                   subject to  b = A*x
#
#  where TV(x) is the 2D total variation of a vectorized version x of
#  an M x N matrix X (i.e., x = vec(X)), b : m x 1 is a vector of
#  measurements (b = A*x), beta > 0, and w_im is a vectorized image
#  similar to the image we want to reconstruct. We use ADMM to solve
#  the above problem, as explained in the attached documentation.
#  Access to A is given implicitly through the function handler  
#  for the operations A*x and A'*y (in arg1 and arg2, respectively). 
#  In this code, A represents the sampling of the images in the Fourier domain. 
#  We consider two cases: 
#   1) Single channel MRI (A_single, AT_single in utils.py). This is selected 
#      by default when the network CRNN is chosen. 
#   2) Mutli-coil MRI (piA, piAt in utils.py). This is selected 
#      by default when the network MoDL is chosen. 
#  * Note that a different operator can be defined in utils to be in line how b was obtained 
#
# Inputs:
#    - M: number of rows of the original image
#    - N: number of columns of the original image (n = M*N)
#    - w_im: n x 1 vector representing an image W_im: w_im = vec(W_im)
#    - b: vector of measurements (of size m)
#    - beta: positive number
#    - mask: sampling mask
#    - nimgs: number of images being processed
#    - rho: penalty of ADMM
#    - max_iter: maximum number of ADMM iterations
#    - csm (optional - only used in multi-coil): coil sensitivity map 
#
# Outputs:
#    - x_opt: solution of the above optimization problem

#  This code was designed and implemented by M. Vella to perform experiments 
#  described in
#  [1] M. Vella, J. F. C. Mota
#      Overcoming Measurement Inconsistency in Deep Learning for Linear Inverse Problems: Applications in Medical Imaging
#      preprint: https://arxiv.org/abs/2011.14387
#      2020

#  Contributors:
#      Marija Vella
#      Joao F. C. Mota
# ========================================================================================
# TVTV_Solver: minimizing TV+TV with linear constraints
# Copyright (C) 2020  Marija Vella
# ========================================================================================
# Any feedback, comments, bugs and questions are welcome and can be sent to mv37@hw.ac.uk
#  =======================================================================================

# ========================================================================================

from numpy import linalg as LA
from utils import supportingFunctions as sf
from pylab import *

# Operation Dx
def D(z, Fc_v, Fc_h):
    D_v = lambda z: np.real(np.fft.ifft(np.multiply(Fc_v, np.fft.fft(z, axis=0)), axis=0))
    D_h = lambda z: np.real(np.fft.ifft(Fc_h * np.fft.fft(z, axis=0), axis=0))
    D = np.concatenate((D_v(z), D_h(z)), axis=0)
    return D

# Operation D'x
def DT(z, Fc_v, Fc_h, n):
    DT_v = lambda z: np.real(np.fft.fft(Fc_v * np.fft.ifft(z, axis=0), axis=0))
    DT_h = lambda z: np.real(np.fft.fft(Fc_h * np.fft.ifft(z, axis=0), axis=0))
    DT = DT_v(z[0:n]) + DT_h(z[n:2*n])
    return DT

# conjugate gradient method
def conjgrad_real(A2, b, x2, A, AT):

    MAX_ITER = 10000
    TOL = 10e-2

    res_p2 = np.expand_dims(A2(x2, A, AT).flatten(),1)
    r = b - np.vstack((np.real(res_p2), np.imag(res_p2)))
    f = r
    rsold = np.dot(np.transpose(r), r)

    for i in range(MAX_ITER):

        f_comp = np.reshape(f, (-1, 2), order='F')
        f_comp = f_comp[:,0] + f_comp[:,1]*1j
        Ap = A2(f_comp, A, AT).flatten()
        Ap = np.expand_dims(Ap, 1)
        Ap = np.vstack((np.real(Ap), np.imag(Ap)))
        alpha = rsold / np.dot(np.transpose(f), Ap)
        x2_real = np.vstack((np.real(x2), np.imag(x2)))
        x2_real = x2_real + alpha * f
        x2 = np.reshape(x2_real, (-1,2), order='F')
        x2 = x2[:, 0] + x2[:, 1]*1j
        x2 = np.expand_dims(x2, 1)
        r = r - alpha * Ap
        rsnew = np.dot(np.transpose(r), r)

        if rsnew < TOL:
           break

        f = r + rsnew/rsold * f
        rsold = rsnew

    return np.expand_dims(x2_real, 1)

# Operation A*A'
def multiply_B(x, A, AT):
    y1 = AT(np.squeeze(x))
    y = A(y1)
    return y


def TVTVSolver(M, N, w_im, beta, b, mask, rho, MAX_ITER, *args):

    for arg in args:
        csm = arg

    w_im = w_im.reshape((-1, 1), order='F')
    n = M * N

    # solver settings
    tau_rho = 10
    mu_rho = 2
    eps_prim = 1e-3
    eps_dual = 1e-3
    m = len(b)

    try:
        csm
        A = lambda z: sf.piA(z, csm, mask, M, N, ncoil=12)
        AT = lambda z: sf.piAt(z, csm, mask, M, N, ncoil=12)
    except:
        A = lambda z: sf.A_single(z, mask, M, N, norm='ortho')
        AT = lambda z: sf.At_single(z, mask, M, N, norm='ortho')

    # Vectors c_h and c_v defining the circulant matrices
    c_h = np.zeros((n, 1))
    c_h[0] = -1
    c_h[n - M] = 1

    c_v = np.zeros((n, 1))
    c_v[0] = -1
    c_v[n - 1] = 1

    # Diagonalazing via FFT
    Fc_h = np.fft.fft(c_h, axis=0)
    Fc_v = np.fft.fft(c_v, axis=0)

    Fc_v_diag = np.real(Fc_v)
    Fc_v_diag_square = Fc_v_diag ** 2  # vector containing diagonal entries squared

    Fc_h_diag = np.real(Fc_h)
    Fc_h_diag_square = Fc_h_diag ** 2  # vector containing diagonal entries squared
    h = 1 / (Fc_v_diag_square + Fc_h_diag_square + 1)

    B = lambda z, A, AT: multiply_B(z, A, AT)

    # Initializing variables for solver
    w_r = D(np.real(w_im), Fc_v, Fc_h)
    w_i = D(np.imag(w_im), Fc_v, Fc_h)
    lam = np.zeros((2*n, 1), dtype=np.complex64)
    mu = np.zeros((n, 1), dtype=np.complex64)
    nita = np.zeros((2*n, 1), dtype=np.complex64)
    c_aux = np.zeros((m, 1))
    r_prim_r = np.zeros((5*n, 1))
    r_prim_i = np.zeros((5*n, 1))
    s_dual_r = np.zeros((5*n, 1))
    s_dual_i = np.zeros((5*n, 1))

    # Splitting into real and imaginary components
    x_r, x_i = np.real(w_im), np.imag(w_im)
    v_r, v_i = x_r, x_i
    u_r, u_i = w_r, w_i
    z_r, z_i = u_r, u_i
    mu_r, mu_i = np.real(mu), np.imag(mu)
    nita_r, nita_i = np.real(nita), np.imag(nita)
    lam_r, lam_i = np.real(lam), np.imag(lam)

    zeta_r = z_r - w_r
    zeta_i = z_i - w_i

    for k in range(MAX_ITER):

        ####### Problem in x #######
        p_r = v_r - (1/rho)*(mu_r)
        p_i = v_i - (1/rho)*(mu_i)
        p = p_r + p_i*1j
        p = p.reshape((M,N), order='F')
        p = p.flatten()

        Ap_all = np.expand_dims(A(p), 1)
        Apb_all = np.vstack((np.real(Ap_all), np.imag(Ap_all))) - np.vstack((np.real(np.expand_dims(b, 1)),
                  np.imag(np.expand_dims(b, 1))))
        c_aux = conjgrad_real(B, Apb_all, c_aux, A, AT)
        c_aux_all = np.squeeze(c_aux).reshape((-1, 2), order='F')
        c_aux_all = c_aux_all[:,0] + c_aux_all[:, 1]*1j
        c_aux = np.expand_dims(c_aux_all, 1)
        c_aux2 = AT(np.squeeze(c_aux_all))
        c_aux2 = np.expand_dims(c_aux2.flatten(), 1)
        c_aux2 = np.vstack((np.real(c_aux2), np.imag(c_aux2)))
        p_final = np.expand_dims(p, 1)
        p_final = np.vstack((np.real(p_final), np.imag(p_final)))
        x2 = p_final - c_aux2

        x2 = np.squeeze(x2).reshape((-1, 2), order='F')
        x_r2 = x2[:,0].reshape((M, N))
        x_r = x_r2.reshape((-1, 1), order='F')
        x_i2 = x2[:,1].reshape((M, N))
        x_i = x_i2.reshape((-1, 1), order='F')
        x = x_r2 + x_i2*1j


        ####### Problem in u #######
        a_r = 0.5*(z_r + D(v_r, Fc_v, Fc_h) - ((lam_r + nita_r)/rho))
        a_i = 0.5*(z_i + D(v_i, Fc_v, Fc_h) - ((lam_i + nita_i)/rho))
        u_r = u_r.copy()
        u_i = u_i.copy()

        # case 1: u_q = 0
        case1 = np.sqrt(a_i**2 + a_r**2) <= 1/(2*rho)
        case1 = case1 * 1
        u_r[np.nonzero(case1)[0]] = 0
        u_i[np.nonzero(case1)[0]] = 0

        # case 2: u_q != 0
        case2 = np.sqrt(a_i**2 + a_r**2) > 1/(2*rho)
        case2 = case2*1
        norma = ((2*rho*np.sqrt(a_r[np.nonzero(case2)[0]]**2 + a_i[np.nonzero(case2)[0]]**2)) - 1) /(2*rho*np.sqrt(a_r[np.nonzero(case2)[0]]**2 + a_i[np.nonzero(case2)[0]]**2))
        u_r[np.nonzero(case2)[0]] = norma * a_r[np.nonzero(case2)[0]]
        u_i[np.nonzero(case2)[0]] = norma * a_i[np.nonzero(case2)[0]]

        ##################################################################################################

        ####### Problem in v #######
        v_prev_r = v_r
        v_prev_i = v_i
        g_r = DT(u_r + (1/rho)*lam_r, Fc_v, Fc_h, n) + (mu_r/rho) + x_r
        g_i = DT(u_i + (1/rho)*lam_i, Fc_v, Fc_h, n) + (mu_i/rho) + x_i
        v_r = np.real(np.fft.ifft(h * np.fft.fft(g_r, axis=0), axis=0))
        v_i = np.real(np.fft.ifft(h * np.fft.fft(g_i, axis=0), axis=0))

        ####### Problem in z #######
        z_prev_r = z_r
        z_prev_i = z_i

        c_r = u_r + (1/rho)*nita_r - w_r
        c_i = u_i + (1/rho)*nita_i - w_i

        # case1 - zeta = 0
        case1z = np.sqrt(c_i**2 + c_r**2) <= beta/(rho)
        case1z = case1z*1
        zeta_r[np.nonzero(case1z)[0]] = 0
        zeta_i[np.nonzero(case1z)[0]] = 0

        # case2 - zeta != 0
        case2z = np.sqrt(c_i ** 2 + c_r ** 2) > beta/(rho)
        case2z = case2z*1
        numerator = (((rho)/beta)*np.sqrt(c_r[np.nonzero(case2z)[0]]**2 + c_i[np.nonzero(case2z)[0]]**2)) - 1
        denominator = ((rho)/beta)*np.sqrt(c_r[np.nonzero(case2z)[0]]**2 + c_i[np.nonzero(case2z)[0]]**2)
        zeta_r[np.nonzero(case2z)[0]] = (numerator/denominator)*c_r[np.nonzero(case2z)[0]]
        zeta_i[np.nonzero(case2z)[0]] = (numerator/denominator)*c_i[np.nonzero(case2z)[0]]

        z_r = zeta_r + w_r
        z_i = zeta_i + w_i

        ##################################################################################################

        # Update dual variables
        lam_r = lam_r + rho*(u_r - D(v_r, Fc_v, Fc_h))
        lam_i = lam_i + rho*(u_i - D(v_i, Fc_v, Fc_h))
        mu_r = mu_r + rho*(x_r - v_r)
        mu_i = mu_i + rho*(x_i - v_i)
        nita_r = nita_r + rho*(u_r - z_r)
        nita_i = nita_i + rho*(u_i - z_i)

        # Primal Residual
        r_prim_r[0:2*n] = u_r - D(v_r, Fc_v, Fc_h)
        r_prim_r[2*n:3*n] = x_r - v_r
        r_prim_r[3*n:5*n] = z_r - u_r

        r_prim_i[0:2*n] = u_i - D(v_i, Fc_v, Fc_h)
        r_prim_i[2*n:3*n] = x_i - v_i
        r_prim_i[3*n:5*n] = z_i - u_i

        r_prim_norm = LA.norm(r_prim_r + r_prim_i*1j, ord=2)

        # Dual Residual
        s_dual_r[0:2*n] = -rho*(D(v_r, Fc_v, Fc_h) + D(v_prev_r, Fc_v, Fc_h))
        s_dual_r[2*n:3*n] = -rho * (v_r + v_prev_r)
        s_dual_r[3*n:5*n] = -rho * (z_r + z_prev_r)

        s_dual_i[0:2*n] = -rho*(D(v_i, Fc_v, Fc_h) + D(v_prev_i, Fc_v, Fc_h))
        s_dual_i[2*n:3*n] = -rho*(v_i + v_prev_i)
        s_dual_i[3*n:5*n] = -rho*(z_i + z_prev_i)

        s_dual_norm = LA.norm(s_dual_r + s_dual_i*1j, ord=2)

        # stopping criteria
        if r_prim_norm < eps_prim and s_dual_norm < eps_dual:
            break

    return x










