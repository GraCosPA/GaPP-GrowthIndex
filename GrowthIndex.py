#===== GaPP kernel for measuring from H(z) data
# Author:
# contact;
# NOTE: the gapp modules only work for python 3.0 atm!

# ======== important packages to be imported 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys as sys

from astropy.cosmology import FlatwCDM, FlatLambdaCDM, Planck15
import astropy.units as u
from gapp import gp, dgp, covariance
import pickle
import numpy as np
import os
import pandas as pd
from numpy import array,concatenate,loadtxt,savetxt,zeros
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import interpolate #interp1d, interp2d, splrep, splev
import multiprocessing
import emcee
import corner

Mpc = 3.085678e22 # m # 1pc = 3.0856776e16 m
c_light = 2.99792458e5
H0_Planck = 67.4 #Planck2018 67.4±0.5 Ωm=0.315±0.007
Om_Planck = 0.315
H0_SH0ES = 73.6 #73.6 \pm 1.1
H0_SH0ES_err = 1.04 #0.5 #1.04
Om_SH0ES = 0.334 #0.018
sigma_80 = 0.8111
sigma_80_err = 0.0060
m_B = 19.253000021067415
convert_other_H0 = False

#covariance.DoubleSquaredExponential，covariance.SquaredExponential, covariance.Matern92, covariance.Matern72, covariance.Matern52, covariance.Matern32
#covfunction=covariance.DoubleSquaredExponential
#covfunction=covariance.SquaredExponential
covfunction=covariance.Matern92

#filename = 'z<0.03_convert'
#filename = 'Growth_Index_SN_fsigma8_SquaredExponential_DLOBS'
#filename = 'Growth_Index_SN_fsigma8_SquaredExponential_DLTH'
#filename = 'Growth_Index_SN_fsigma8_SquaredExponential_PL'
filename = 'Growth_Index_SN_fsigma8_Matern92_DLOBS'
#filename = 'Growth_Index_SN_fsigma8_Matern92_DLTH'
 
if __name__=="__main__":

    #======== loading SN data ==========
    # Default is to use SN data from https://arxiv.org/abs/2202.04077
    # and Cepheids data from https://arxiv.org/abs/2112.04510
    default_sn_data_file = 'Pantheon+SH0ES.dat'
    default_sn_covmat_file = 'Pantheon+SH0ES_STAT+SYS.cov'
    
    #======== loading CC data ==========
    default_Hz_data_file = 'hz_cc_sdss.dat'
    #default_Hz_data_file = 'hz_cc_sdss_sn.dat'
    print("Loading HZ Data Point From {}".format(default_Hz_data_file))
    (Z,Hz,Sigma,hid) = loadtxt(default_Hz_data_file,unpack='True')
    print('Loading HZ Data Done ... ')
    
    print("Loading SN Data From {}".format(default_sn_data_file))
    sndata = pd.read_csv(default_sn_data_file,delim_whitespace=True)
    snoriglen = len(sndata)
    # Selection data sets #USED_IN_SH0ES_HF - 1 if used in SH0ES 2021 Hubble Flow dataset. 0 if not included. z ∈ [0.00122, 2.26137]
    #ww = (sndata['zHD']>0.01)
    #ww = (sndata['zHD']>0.0)
    #ww = (sndata['zHD']>0.0) & (sndata['USED_IN_SH0ES_HF']==0)
    #ww = (sndata['zHEL']>0.0) & (sndata['USED_IN_SH0ES_HF']==0)
    ww = (sndata['zHD']>0.0) & (sndata['USED_IN_SH0ES_HF']==0)
    #ww = (sndata['zHD']<0.03) & (sndata['USED_IN_SH0ES_HF']==0) #(np.array(sndata['IS_CALIBRATOR'],dtype=bool))
        
    zHD = sndata['zHD'][ww] #Hubble Diagram Redshift (with CMB and VPEC corrections)
    zHD_err = sndata['zHDERR'][ww] #Hubble Diagram Redshift Uncertainty
    #print(type(zHD))
    #print('zHD=',zHD.values)
    zHEL = sndata['zHEL'][ww] #Heliocentric Redshift
    zHELERR = sndata['zHELERR'][ww] #Heliocentric Redshift Uncertainty
    zCMB = sndata['zCMB'][ww] #Heliocentric Redshift
    zCMBERR = sndata['zCMBERR'][ww] #Heliocentric Redshift Uncertainty
    m_obs = sndata['m_b_corr'][ww] #Tripp1998 corrected/standardized m_b magnitude
    m_b_corr_err = sndata['m_b_corr_err_DIAG'][ww] #Tripp1998 corrected/standardized m_b magnitude uncertainty as determined from the diagonal of the covariance matrix. **WARNING, DO NOT FIT COSMOLOGICAL PARAMETERS WITH THESE UNCERTAINTIES. YOU MUST USE THE FULL COVARIANCE. THIS IS ONLY FOR PLOTTING/VISUAL PURPOSES**
    MU_SH0ES_obs = sndata['MU_SH0ES'][ww] #Tripp1998 corrected/standardized distance modulus where fiducial SNIa magnitude (M) has been determined from SH0ES 2021 Cepheid host distances.
    MU_SH0ES_obs_ERR_DIAG = sndata['MU_SH0ES_ERR_DIAG'][ww] #Uncertainty on MU_SH0ES as determined from the diagonal of the covariance matrix. **WARNING, DO NOT FIT COSMOLOGICAL PARAMETERS WITH THESE UNCERTAINTIES. YOU MUST USE THE FULL COVARIANCE. THIS IS ONLY FOR PLOTTING/VISUAL PURPOSES**
    #mB = sndata['mB'][ww] #SALT2 uncorrected brightness
    #mB_err = sndata['mBERR'][ww] #SALT2 uncorrected brightness uncertainty
        
    zSN=zHD.values
    zSN_err=zHD_err.values
    
    m_obs_SN=m_obs.values
    m_b_corr_err_SN = m_b_corr_err.values
    mu_obs_SN_SH0ES = MU_SH0ES_obs.values
    mu_obs_err_SN_SH0ES = MU_SH0ES_obs_ERR_DIAG.values
    print('Loading',snoriglen,'SN Data Done.',len(zSN),' Used ...')
    
    #plt.plot(zSN,m_obs_SN, linestyle=(0, (3, 5, 1, 5, 1, 5)),linewidth=1.5, color='red',label=r"m_obs_SN")
    #plt.plot(zSN,mu_obs_SN_SH0ES-m_B, linestyle=(0, (3, 5, 1, 5, 1, 5)),linewidth=1.5, color='blue',label=r"mu_obs_SN_SH0ES")
    #plt.plot(zSN,mB, linestyle=(0, (3, 5, 1, 5, 1, 5)),linewidth=1.5, color='green',label=r"mB")
    #plt.show()
    #sys.exit()
    
    # Delta H0 -> Delta mu -5.0*Delta H_0/(H_0*ln(10.0))
    # Convert to other H_0 values
    if (convert_other_H0):
       mu_obs_SN_SH0ES = mu_obs_SN_SH0ES-5.0*(H0_Planck-H0_SH0ES)/(H0_SH0ES*np.log(10.0))
       H0_SH0ES = H0_Planck
       Om_SH0ES = Om_Planck
    
    DL_SN = 10.0**((mu_obs_SN_SH0ES-25.0)/5.0)
    DC_obs_SN = DL_SN/(1.0+zSN)
    DC_corr_err_SN = (DL_SN/(1.0+zSN)**2)**2*zSN_err**2 + ( np.log(10.0)/5.0*(DL_SN/(1+zSN)))**2 * mu_obs_err_SN_SH0ES**2
    DC_corr_err_SN = np.sqrt(DC_corr_err_SN)

    cosmology_th = FlatLambdaCDM(H0=H0_SH0ES, Om0=Om_SH0ES,)
    DL_th = cosmology_th.luminosity_distance(zSN).to(u.Mpc).value
    DC_th = DL_th/(1.0+zSN)
    #print(DC_th/DC_obs_SN)
      
    #DL_SN = DL_th
      
    print("Loading SN Covariance from {}".format(default_sn_covmat_file))
    # The file format for the covariance has the first line as an integer
    # indicating the number of covariance elements, and the the subsequent
    # lines being the elements.
    # This function reads in the file and the nasty for loops trim down the covariance
    # to match the only rows of data that are used for cosmology

    f = open(default_sn_covmat_file)
    line = f.readline()
    n_cov = len(zSN)
    C_cov = np.zeros((n_cov,n_cov))
    ii = -1
    jj = -1
    mine = 999
    maxe = -999
    for i in range(snoriglen):
        jj = -1
        if ww[i]:
            ii += 1
        for j in range(snoriglen):
            if ww[j]:
                jj += 1
            val = float(f.readline())
            if ww[i]:
                if ww[j]:
                    C_cov[ii,jj] = DL_SN[ii]/(1.0+zSN[ii])*np.log(10.0)/5.0 * val * DL_SN[jj]/(1.0+zSN[jj])*np.log(10.0)/5.0
                    if (ii == jj):
                       C_cov[ii,ii] = C_cov[ii,ii] + (DL_SN[ii]/(1.0+zSN[ii])**2)**2*zSN_err[ii]**2
    f.close()
    print('Loading SN Cov Matrix Done ...')
    
#################### f \sigma_8 ############################
    default_GRW_data_file = 'Growth_tableII.txt'
    default_GRWiggle_data_file = 'Cij_WiggleZ.txt'
    print("Loading f\sigma_8 Data Point From {}".format(default_GRW_data_file))
    print("Loading Wiggle f\sigma_8 Data Point From {}".format(default_GRWiggle_data_file))
    fs8dat = np.loadtxt(default_GRW_data_file, unpack=True)
    fs8cov_w = np.loadtxt(default_GRWiggle_data_file, unpack=True)
    
    n_fs8 = fs8dat.shape[1]
    z_fs8 = fs8dat[0]
    fs8_obs = fs8dat[1]
    fs8_sig = fs8dat[2]
    fs8_Om0 = fs8dat[3]
    Cij_fs8 = np.eye(n_fs8)* fs8_sig**2
    Cij_fs8[9:12,9:12] = fs8cov_w
    
    zmin = np.min([np.min(zSN),np.min(Z),np.min(z_fs8)])
    zmax = np.max([np.max(zSN),np.max(Z),np.max(z_fs8)])
    
    f_rec_data = np.zeros((len(zSN),3))
    f_rec_data[:,0] = zSN[:]
    f_rec_data[:,1] = DC_obs_SN[:]
    f_rec_data[:,2] = DC_corr_err_SN[:]
    savetxt("f_rec_data_"+filename+".txt",f_rec_data)
    
    DX = Z
    #Hz = cosmology_th.H(Z).value
    DY = c_light/Hz
    DSigma = np.sqrt( (c_light/Hz**2)**2*Sigma**2 )

    df_rec_data = np.zeros((len(Z),3))
    df_rec_data[:,0] = DX[:]
    df_rec_data[:,1] = DY[:]
    df_rec_data[:,2] = DSigma[:]
    savetxt("df_rec_data_"+filename+".txt",df_rec_data)
    
    print("Starting SN+OHD Gaussian Process ...")
    #g = dgp.DGaussianProcess(zSN,DC_obs_SN,C_cov,dX=DX, dY=DY, dSigma=DSigma,covfunction=covfunction,cXstar=(zmin,zmax,1000), grad='False')
    
    g = dgp.DGaussianProcess(zSN,DC_obs_SN,C_cov,dX=DX, dY=DY, dSigma=DSigma,covfunction=covfunction,cXstar=(zmin,zmax,1000), grad='False')
    #g = dgp.DGaussianProcess(zSN,DC_obs_SN,C_cov,covfunction=covfunction,cXstar=(zmin,zmax,1000),grad='False')
    
    (dcrec, theta) = g.gp(thetatrain='True')
    # reconstruction of the first, second, third and forth derivatives.
    # theta is fixed to the previously determined value.
    (dcdrec, theta) = g.dgp(thetatrain='True')
    (dcd2rec, theta) = g.d2gp(thetatrain='True')
        
    print("Gaussian SN+OHD Process Finished ...")
    
    # save the output
    savetxt(filename+"_fdc.txt", dcrec)
    savetxt(filename+"_dfdc.txt", dcdrec)
    savetxt(filename+"_d2fdc.txt", dcd2rec)
    ###################### Covariance Metrix HD/HD_f fsigma_8 ####################
    
    ZZ = np.zeros((len(dcrec[:, 0])))
    ZZ = dcrec[:, 0]
    
    DC = np.zeros((len(dcrec[:, 0])))
    DDC = np.zeros((len(dcrec[:, 0])))
    DC[:] = dcrec[:, 1]
    DDC[:] = dcrec[:, 2]
    
    DC1 = np.zeros((len(dcdrec[:, 0])))
    DDC1 = np.zeros((len(dcdrec[:, 0])))
    DC1[:] = dcdrec[:, 1]
    DDC1[:] = dcdrec[:, 2]

    HD = c_light/DC1 * DC / (1.0+ZZ)
    fHD = interpolate.interp1d(ZZ,HD,kind='cubic')
    
    dHD = - c_light*DC*DDC1/DC1**2/ (1.0+ZZ) + c_light/DC1 * DDC / (1.0+ZZ)
    fdHD = interpolate.interp1d(ZZ,dHD,kind='cubic')
    
    cosmology_th_SN = FlatLambdaCDM(H0=H0_SH0ES, Om0=Om_SH0ES,)
    HDA_th = cosmology_th_SN.H(z_fs8).value*cosmology_th_SN.angular_diameter_distance(z_fs8).to(u.Mpc).value
    HDA_fid = np.zeros(n_fs8)
    
    for i in range(n_fs8):
        cosmology_th_fid = FlatLambdaCDM(H0=H0_SH0ES, Om0=fs8_Om0[i],)
        HDA_fid[i] = cosmology_th_fid.H(z_fs8[i]).value*cosmology_th_fid.angular_diameter_distance(z_fs8[i]).to(u.Mpc).value
        
    ratio_AP = HDA_th/HDA_fid
    fs8_obs_fid = ratio_AP * fs8_obs
    Cij_fs8_fid = ratio_AP**2 * Cij_fs8
    for i in range(9,12):
        for j in range(9,i+1):
                Cij_fs8_fid[i,j] = ratio_AP[i] * ratio_AP[j] * fs8cov_w[i-9, j-9]
                Cij_fs8_fid[j,i] = Cij_fs8_fid[i,j]

    for i in range(n_fs8):
        for j in range(n_fs8):
            Cij_fs8_fid[i,j] = Cij_fs8_fid[i,j] + fs8_obs_fid[i]/HDA_fid[i] * fdHD(ZZ[i]) * fdHD(ZZ[j]) * fs8_obs_fid[j]/HDA_fid[j]
    
    print("Starting f\sigma_8 Gaussian Process ...")
    gfs8 = dgp.DGaussianProcess(z_fs8,fs8_obs_fid,Cij_fs8_fid,covfunction=covfunction,cXstar=(zmin,zmax,1000), grad='False')
                
    (fs8rec, theta) = gfs8.gp(thetatrain='True')
    # reconstruction of the first, second, third and forth derivatives.
    # theta is fixed to the previously determined value.
    (fs8drec, theta) = gfs8.dgp(thetatrain='True')
     
    print("Gaussian f\sigma_8  Process Finished ...")
    # save the output
    savetxt(filename+"_fs8.txt", fs8rec)
    savetxt(filename+"_dfs8.txt", fs8drec)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12,9))
    fontsize = 15
    plt.xlim(zmin, zmax)
    plt.plot(fs8rec[:, 0], fs8rec[:, 1])
    plt.fill_between(fs8rec[:,0], fs8rec[:,1] + fs8rec[:,2],
                             fs8rec[:,1] - fs8rec[:,2], facecolor='lightblue')
    plt.errorbar(z_fs8, fs8_obs_fid, np.sqrt(np.diagonal(Cij_fs8_fid)), color='red', fmt='o')
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$f\sigma_8$', fontsize=fontsize)
    plt.legend(("Rec $f\sigma_8$","Rec $f\sigma_8\pm 1\sigma$", "$f\sigma_{8_{\r obs}}$"), fontsize=fontsize, loc='upper right')
    plt.savefig(filename+'plot_fs8.pdf')
    
    fs8obs = np.zeros((n_fs8,3))
    fs8obs[:,0] = z_fs8
    fs8obs[:,1] = fs8_obs_fid
    fs8obs[:,2] = np.sqrt(np.diagonal(Cij_fs8_fid))
    savetxt(filename+"_fsigma8obs.txt", fs8obs)
    #sys.exit()
    

    # test if matplotlib is installed
    try:
        import matplotlib.pyplot
    except:
        print("matplotlib not installed. no plots will be produced.")
        exit
    # create plot
    import plotdcsgrowth#, plotModifiedGravity
    plotdcsgrowth.plot(zSN, DC_obs_SN, DC_corr_err_SN, DX, DY, DSigma, dcrec, dcdrec, dcd2rec, fs8rec, fs8drec, fs8obs, filename=filename)
    #plotModifiedGravity.plot(zSN, DC_obs_SN, DC_corr_err_SN, DX, DY, DSigma, dcrec, dcdrec, dcd2rec, fs8rec, fs8drec, fs8obs, filename=filename)
