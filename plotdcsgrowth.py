import matplotlib.pyplot as plt
from numpy import loadtxt, savetxt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import sys as sys
from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative

Mpc = 3.085678e22 # m # 1pc = 3.0856776e16 m
c_light = 2.99792458e5
H0_Planck = 67.4 #Planck2018 67.4±0.5 Ωm=0.315±0.007
Om_Planck = 0.315
H0_SH0ES = 73.6 #73.6 \pm 1.1
H0_SH0ES_err = 1.04 #0.5 #1.04
Om_SH0ES = 0.27 #0.334 #0.018
Om_SH0ES_err = 0.018
sigma_80 = 0.8111
sigma_80_err = 0.0060

def plot(X, Y, Sigma, dX, dY, dSigma, rec, drec, d2rec, fs8rec, dfs8rec, fs8obs, xmin=None,xmax=None,filename=''):

    default_sn_data_file = 'Pantheon+SH0ES.dat'
    default_sn_covmat_file = 'Pantheon+SH0ES_STAT+SYS.cov'
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

    zSN=zHD.values
    zSN_err=zHD_err.values
    
        #======== loading CC data ==========
    default_Hz_data_file = 'hz_cc_sdss.dat'
    #default_Hz_data_file = 'hz_cc_sdss_sn.dat'
    print("Loading HZ Data Point From {}".format(default_Hz_data_file))
    (zCC,HzCC,SigmaCC,hid) = loadtxt(default_Hz_data_file,unpack='True')
    print('Loading HZ Data Done ... ')
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.figure(figsize=(12,9))
    
    fontsize = 10
    if (xmin==None):
       xmin = np.min(rec[:, 0])
    else:
       xmin = xmin
    if (xmax==None):
       xmax = np.max(rec[:, 0])
    else:
       xmax = xmax
       
    Z = np.zeros((len(rec[:, 0])))
    Z = rec[:, 0]
    
    DC = np.zeros((len(rec[:, 0])))
    DC_err = np.zeros((len(rec[:, 0])))
    DC[:] = rec[:, 1]
    DC_err[:] = rec[:, 2]
    
    DC1 = np.zeros((len(drec[:, 0])))
    DC1_err = np.zeros((len(drec[:, 0])))
    DC1[:] = drec[:, 1]
    DC1_err[:] = drec[:, 2]
    
    DC2 = np.zeros((len(d2rec[:, 0])))
    DC2_err = np.zeros((len(d2rec[:, 0])))
    DC2[:] = d2rec[:, 1]
    DC2_err[:] = d2rec[:, 2]
    
    Oml_SH0ES = 1.0 - Om_SH0ES

    h_lcdm = H0_SH0ES*np.sqrt(Om_SH0ES*(1. + Z)**3 + Oml_SH0ES)
    
    Hz = c_light/DC1
    Hz_err = c_light/DC1**2*DC1_err
    
    recomz = np.zeros((len(rec[:, 0])))
    recomz = Om_SH0ES * (1.+Z)**3 * DC1**2 * H0_SH0ES**2 /c_light**2
    #for i in range(len(Z)):
    #    if ((recomz[i] >= 1.0-1e-3) & (recomz[i] < 1.0+1e-3)):
    #       print(Z[i])
    #sys.exit()
    
    sigma_omz = (H0_SH0ES**2*Om_SH0ES_err*(1.+Z)**3/Hz**2)**2 + (2.*Om_SH0ES * H0_SH0ES*H0_SH0ES_err*(1.+Z)**3/Hz**2)**2 + (2.*Om_SH0ES * H0_SH0ES**2*(1+Z)**3*Hz_err/Hz**3)**2
    sigma_omz = np.sqrt(sigma_omz)
    
    # f\sigma_8
    fs8 = np.zeros((len(rec[:, 0])))
    fs8 = fs8rec[:,1]
    fs8_err = np.zeros((len(rec[:, 0])))
    fs8_err = fs8rec[:,2]
    
    # f\sigma_8'
    d_fs8_dz =np.zeros((len(rec[:, 0])))
    d_fs8_dz = dfs8rec[:,1]
    d_fs8_dz_err = np.zeros((len(rec[:, 0])))
    d_fs8_dz_err = dfs8rec[:,2]
    
    delta_p_over_delta0 = -1./sigma_80 * fs8/(1.+Z)
    delta_p_over_delta0_err = (sigma_80_err/sigma_80**2 * fs8/(1.+Z))**2 + (1./sigma_80 * fs8_err/(1.+Z))**2
    delta_p_over_delta0_err = np.sqrt(delta_p_over_delta0_err)
        
    delta_over_delta0 = np.zeros((len(rec[:, 0])))
    delta_over_delta0[0] = 1.0
    delta_over_delta0[1:len(rec[:, 0])] = 1. + cumtrapz(delta_p_over_delta0, Z)
    delta_over_delta0_err = ((delta_over_delta0 - 1.)*sigma_80_err/sigma_80)**2 + ((delta_over_delta0 - 1.)*fs8_err/(1.+Z))**2
    delta_over_delta0_err = np.sqrt(delta_over_delta0_err)
    
    delta_p2_over_delta0 = np.zeros((len(rec[:, 0])))
    delta_p2_over_delta0 = 1./sigma_80 * fs8/(1.+Z)**2 - 1./sigma_80 * d_fs8_dz /(1.+Z)
    delta_p2_over_delta0_err = (1./sigma_80 * fs8_err/(1.+Z)**2)**2 + (1./sigma_80 * d_fs8_dz_err /(1.+Z))**2
    delta_p2_over_delta0_err = np.sqrt(delta_p2_over_delta0_err)
    # Check the same, done!
    #dz = Z[1] - Z[0]
    #delta_p2_over_delta0_func = np.gradient(delta_p_over_delta0, dz)
    #plt.plot(Z,delta_over_delta0,color='black')
    #plt.plot(Z,delta_p2_over_delta0,color='red')
    #plt.plot(Z,delta_p2_over_delta0_func,color='blue')
    #plt.show()
    #sys.exit()
    #########################  f_growth(z) #########################
    f_growth = -(1.+Z) * delta_p_over_delta0/delta_over_delta0
    f_growth_err = ((1.+Z)*delta_p_over_delta0_err/delta_over_delta0)**2 + ((1.+Z)*delta_p_over_delta0*delta_over_delta0_err/delta_over_delta0**2)**2
    f_growth_err = np.sqrt(f_growth_err)
    #########################  d f_growth(z) / dz #########################
    # derivative of f(z) with respect to z
    #dz = Z[1] - Z[0]
    #df_growth_dz = np.gradient(f_growth, dz)
    # Get a function that evaluates the linear spline at any x
    #fgrowth = InterpolatedUnivariateSpline(Z, f_growth, k=1)
    # Get a function that evaluates the derivative of the linear spline at any x
    #dfdz = fgrowth.derivative()
    # Evaluate the derivative dydx at each x location...
    #df_growth_dz = dfdz(Z)
    df_growth_dz = - delta_p_over_delta0/delta_over_delta0 - (1. + Z) * ( (delta_p2_over_delta0/delta_over_delta0 - delta_p_over_delta0**2/delta_over_delta0**2))
    df_growth_dz_err = ( delta_p_over_delta0_err / delta_over_delta0 )**2  \
                      + (delta_p_over_delta0 * delta_over_delta0_err/delta_over_delta0**2)**2    \
                      + (1.+Z)**2 * (delta_p2_over_delta0_err/delta_over_delta0 )**2    \
                      + (1.+Z)**2 * (delta_p2_over_delta0*delta_over_delta0_err/delta_over_delta0**2 )**2  \
                      + (1.+Z)**2 * (2.*delta_p_over_delta0/delta_over_delta0)**2 * (delta_p_over_delta0_err / delta_over_delta0)**2 \
                      + (1.+Z)**2 * (2.*delta_p_over_delta0/delta_over_delta0)**2 * (delta_p_over_delta0 * delta_over_delta0_err/delta_over_delta0**2)**2
    df_growth_dz_err = np.sqrt(df_growth_dz_err)
    # f\sigma_8' and np.gradient(fs8, dz) are the same
    #dz = Z[1] - Z[0]
    #d_fs8_dz_func = np.gradient(fs8, dz)
    #plt.plot(Z,d_fs8_dz,color='red')
    #plt.plot(Z,d_fs8_dz_func,color='blue')
    #plt.show()
    #sys.exit()
    #plt.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.95,hspace=0.35,wspace=0.25)
    plt.subplot(2,1,1)
    plt.xlim(xmin, xmax)
    plt.plot(Z,(Om_SH0ES * (1.+Z)**3/(Om_SH0ES * (1.+Z)**3 + 1.-Om_SH0ES))**(6./11.),linestyle=(0, (3, 5, 1, 5, 1, 5)),linewidth=1.5, color='red',label=r"$\Lambda$CDM $f(z)=\Omega^{\gamma}_{m}(z)$")
    plt.plot(Z,f_growth,linewidth=1.5, color='blue',label=r"$f(z)$")
    plt.fill_between(Z, f_growth + f_growth_err, f_growth - f_growth_err, facecolor='blue',alpha=0.2,label=r"$f(z)\pm 1\sigma$")
    plt.fill_between(Z, f_growth + 2.*f_growth_err, f_growth - 2.*f_growth_err, facecolor='lightblue',alpha=0.4,label=r"$f(z) \pm 2\sigma$")
    plt.fill_between(Z, f_growth + 3.*f_growth_err, f_growth - 3.*f_growth_err, facecolor='lightblue',alpha=0.6,label=r"$f(z) \pm 3\sigma$")
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$f(z)$', fontsize=fontsize)
    plt.legend(fontsize='10', loc='upper left')
    
    plt.subplot(2,1,2)
    plt.xlim(xmin, xmax)
    plt.plot(Z,df_growth_dz,linewidth=1.5, color='blue',label=r"$df(z)/dz$")
    plt.fill_between(Z, df_growth_dz + df_growth_dz_err, df_growth_dz - df_growth_dz_err, facecolor='blue',alpha=0.2,label=r"$df(z)/dz \pm 1\sigma$")
    plt.fill_between(Z, df_growth_dz + 2.*df_growth_dz_err, df_growth_dz - 2.*df_growth_dz_err, facecolor='lightblue',alpha=0.4,label=r"$df(z)/dz \pm 2\sigma$")
    plt.fill_between(Z, df_growth_dz + 3.*df_growth_dz_err, df_growth_dz - 3.*df_growth_dz_err, facecolor='lightblue',alpha=0.6,label=r"$df(z)/dz \pm 3\sigma$")
    plt.hlines(0.,xmin, xmax,colors='k',linestyles='dashed',alpha=0.4)
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$df(z)/dz$', fontsize=fontsize)
    plt.legend(fontsize='10', loc='lower left')
    #plt.show()
    #sys.exit()
    plt.savefig(filename+'plot_MG_f_growth.pdf')
    plt.close()

    
    gammas_orig = np.log(f_growth)/np.log(recomz)
    gammas_err = (f_growth_err/f_growth/np.log(recomz))**2 + (np.log(f_growth)/np.log(recomz)**2*sigma_omz/recomz)**2
    
    gammas_err = np.sqrt(gammas_err)
    gammas = gammas_orig
    
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
        
    gammassmooth = smooth(gammas_orig, 400)
    
    for i in range(len(Z)):
        if (np.abs(gammassmooth[i]-gammas_orig[i])<1e-3):
           print('smooth at:',Z[i])
        if (np.abs(recomz[i]-1.0)<1e-3):
           print('Omega_m=1 at:',Z[i])
           zomone = Z[i]
    #sys.exit()
    #for i in range(len(Z)):
    #    if ((Z[i] >= zomone-0.5) & (Z[i] < zomone+0.5)):
    #        #print(i)
    #        gammas[i] = gammassmooth[i]
    #        gammas_err[i] = (gammas_err[550]+gammas_err[824])/2.0
    ######################## MG mu ########################
    
    recmu = 2./(3. * recomz) *( -(1.+Z)*df_growth_dz + f_growth**2 + (2. + (1.+Z)*DC2/DC1)*f_growth )
    
    recmu_err = (2.* sigma_omz/3. /recomz**2)**2 *( -(1.+Z)*df_growth_dz + f_growth**2 + (2. + (1.+Z)*DC2/DC1)*f_growth )**2 \
              + (2./(3. * recomz))**2 * ( (1.+Z)**2*df_growth_dz_err**2 + (2*f_growth * f_growth_err)**2 \
              +  ((1.+Z)*(DC2_err/DC1) * f_growth )**2 +  ( (1.+Z)*(DC2*DC1_err/DC1**2) * f_growth )**2 \
              + ((2. + (1.+Z)*DC2/DC1)*f_growth_err)**2 )
    recmu_err = np.sqrt(recmu_err)
    #mu_DGP = 1+1/3\beta
    mu_DGP = (2. + 4.* (Om_SH0ES * (1.+Z)**3 )**2)/(3.*(1. + (Om_SH0ES * (1.+Z)**3 )**2))
    # 0.343 -1.200\pm 1.025
    # 1 -0.944  \pm 0.253
    # 2 -1.156  \pm 0.341
    # 3 -1.534  \pm 0.453
    # 4 -2.006  \pm 0.538
    # 5 -2.542  \pm 0.689
    # 6 -3.110  \pm 0.77
    n = 1
    g_a = -0.944
    mu_par1 = 1. + g_a*(Z/(1.+Z))**n - g_a*(Z/(1.+Z))**(2*n)
    
    n = 2
    g_a = -1.156
    mu_par2 = 1. + g_a*(Z/(1.+Z))**n - g_a*(Z/(1.+Z))**(2*n)
    
    plt.xlim(xmin, xmax)
    plt.plot(Z,recmu,color='blue',label=r"Rec $\mu$")
    plt.fill_between(Z, recmu + recmu_err, recmu - recmu_err, facecolor='blue',alpha=0.2,label=r"Rec $\mu\pm 1\sigma$")
    plt.fill_between(Z, recmu + 2.*recmu_err, recmu - 2.*recmu_err, facecolor='lightblue',alpha=0.4,label=r"Rec $\mu\pm 2\sigma$")
    plt.fill_between(Z, recmu + 3.*recmu_err, recmu - 3.*recmu_err, facecolor='lightblue',alpha=0.6,label=r"Rec $\mu\pm 3\sigma$")
    plt.plot(Z,mu_DGP,'g,',label=r"$\mu_{\rm DGP}=1+\frac{1}{3\beta}$")
    plt.plot(Z,mu_par1,'k-.',label=r"$\mu_{\rm P}=1+g_a(\frac{1}{1+z})^n - g_a(\frac{1}{1+z})^{2n}$, $n=1,g_a=-0.944$")
    plt.plot(Z,mu_par2,'b:',label=r"$\mu_{\rm P}=1+g_a(\frac{1}{1+z})^n - g_a(\frac{1}{1+z})^{2n}$, $n=2,g_a=-1.156$")
    plt.hlines(1,xmin, xmax,colors='red',linestyles='dashed',alpha=0.6, label=r"$\mu_{\rm GR}\equiv 1$")
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$\mu(z)=\\frac{G_{\\rm eff}}{G_{\\rm N}}$', fontsize=fontsize)
    plt.legend(fontsize='4.9', loc='lower left')
    #plt.show()
    #sys.exit()
    plt.savefig(filename+'plot_MG_mu.pdf')
    plt.close()
        
    qz = -(1.0+Z)*DC2/DC1 - 1.0
    qz_err = ((1.+Z)*DC2/DC1**2)**2*DC1_err**2 + ((1.+Z)/DC1)**2*DC2_err**2
    qz_err = np.sqrt(qz_err)
    #plt.plot(Z,qz,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='blue',label=r"$q(z)$")
    #plt.show()
    #sys.exit()
    #RicciScalar = 6.*Hz**2 *(2. + (1.+Z)*DC2/DC1)
    RicciScalar = 6.*Hz**2 *(1. - qz)
    q0=qz[0]
    RicciScalar0 = 6.*H0_SH0ES**2*(1-q0)
    
    #plt.plot(Z,RicciScalar,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='blue',label=r"$R$")
    #plt.show()
    #sys.exit()
    
    
    BigF = 1./recmu

    dz = Z[1] - Z[0]
    dBigF_dz = np.gradient(BigF, dz)
    
    plt.plot(Z,BigF,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='black',label=r"$F=\partial f(R)/\partial R$")
    plt.plot(Z,dBigF_dz,linewidth=1.5, color='red',label=r"F'")
    plt.plot(Z,1.-qz - BigF + (1.+Z)*dBigF_dz+recomz,linewidth=1.5, color='blue',label=r"[...]")
    plt.plot(Z,recomz,linewidth=1.5, color='blue',label=r"$\Omega_{m}(z)$")
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$F,F\'$', fontsize=fontsize)
    plt.legend(fontsize='10', loc='upper left')
    #plt.show()
    plt.savefig(filename+'plot_MG_FOm.pdf')
    #sys.exit()
    plt.close()
    
    f_gravity = BigF*RicciScalar - 6.*BigF*Hz**2 + 6.*(1.+Z)*Hz**2*dBigF_dz + 6.*Hz**2*recomz
    alpha = 0.21
    beta = 0.12
    gamma = 0.14
    f_exampe = RicciScalar + alpha*RicciScalar0**2/RicciScalar + beta*RicciScalar0**3/RicciScalar**2 + gamma*RicciScalar0**4/RicciScalar**3  #*(RicciScalar/RicciScalar0)**(2*n)/( (RicciScalar/RicciScalar0)**(2*n) + 1.)
    #plt.plot(Z,RicciScalar,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='red',label=r"$R$")
    #plt.plot(Z,f_gravity,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='blue',label=r"$f(R)$")
    plt.plot(RicciScalar,f_gravity,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='blue',label=r"$f(R)$")
    plt.plot(RicciScalar,f_exampe,linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=1.5, color='black',label=r"$f(R)$")
    plt.plot(RicciScalar,RicciScalar,linewidth=1.5, color='red',label=r"GR")
    plt.xlabel('$R$', fontsize=fontsize)
    plt.ylabel('$f(R)$', fontsize=fontsize)
    plt.legend(fontsize='10', loc='upper left')
    #plt.show()
    #sys.exit()
    plt.savefig(filename+'plot_MG_fR.pdf')
    plt.close()
   
    plt.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.95,hspace=0.35,wspace=0.25)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.figure(figsize=(12,9))
    
    fontsize = 10
    plt.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.95,hspace=0.35,wspace=0.25)

    plt.subplot(3,2,1)
    plt.xlim(xmin, xmax)
    #plt.ylim(0, 1.5e4)
    #plt.xscale('log')
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                     facecolor='lightblue',alpha=0.9)
    plt.plot(rec[:, 0], rec[:, 1],linewidth=1.5, color='blue')
    plt.errorbar(X[0:1424], Y[0:1424], Sigma[0:1424], color='red', fmt='o')
    plt.errorbar(X[1425:], Y[1425:], Sigma[1425:], color='blue', fmt='o')
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$D_C(z)$', fontsize=fontsize)
    plt.legend(("Rec $1\sigma$","Rec", "Pantheon+"), fontsize='10', loc='upper left')
    
    """
    plt.subplot(3,2,2)
    plt.xlim(xmin, xmax)
    #plt.xscale('log')
    plt.fill_between(drec[:, 0], drec[:, 1] + drec[:, 2],
                     drec[:, 1] - drec[:, 2], facecolor='lightblue')
    plt.plot(drec[:, 0], drec[:, 1])
    plt.errorbar(dX, dY, dSigma, color='red', fmt='o')
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$D^{\'}_C(z)$', fontsize=fontsize)
    plt.legend(("Rec $1\sigma$","Rec", "CC+BAO"), fontsize='10', loc='upper right')
    """
    plt.subplot(3,2,2)
    plt.xlim(xmin, xmax)
    #plt.xscale('log')
    plt.fill_between(Z, Hz + Hz_err, Hz - Hz_err, facecolor='lightblue')
    plt.plot(Z, Hz,linewidth=1.5, color='blue')
    plt.plot(Z, h_lcdm,linestyle=(5, (10, 3)), linewidth=1.5, color='black')
    plt.errorbar(zCC,HzCC,SigmaCC, color='red', fmt='o')
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$H(z)$', fontsize=fontsize)
    plt.legend(("Rec $1\sigma$","Rec","$H(z)$ $\Lambda$CDM","CC+BAO"), fontsize='10', loc='upper left')
    
    plt.subplot(3,2,3)
    plt.xlim(xmin, xmax)
    #plt.xscale('log')
    #plt.fill_between(d4rec[:, 0], d4rec[:, 1] + d4rec[:, 2],
    #                 d4rec[:, 1] - d4rec[:, 2], facecolor='lightblue')
    #plt.plot(Z, f_growth)
    plt.plot(fs8rec[:, 0], fs8rec[:, 1],linewidth=1.5, color='blue')
    plt.fill_between(fs8rec[:,0], fs8rec[:,1] + fs8rec[:,2],
                             fs8rec[:,1] - fs8rec[:,2], facecolor='lightblue')
    plt.errorbar(fs8obs[:,0],fs8obs[:,1],fs8obs[:,2], color='red', fmt='o')
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$f\sigma_8(z)$', fontsize=fontsize)
    plt.legend(("Rec", "Rec $1\sigma$","$f\sigma_8(z)_{obs}$"), fontsize='10', loc='upper right')
    
    plt.subplot(3,2,4)
    plt.xlim(xmin, xmax)
    plt.plot(Z,(Om_SH0ES * (1.+Z)**3/(Om_SH0ES * (1.+Z)**3 + 1.-Om_SH0ES))**(6./11.),linestyle=(5, (10, 3)), linewidth=1.5, color='black',label=r"$\Lambda$CDM $f(z)=\Omega^{\gamma}_{m}(z)$")
    plt.plot(Z,f_growth,linewidth=1.5, color='blue',label=r"$f(z)$")
    plt.fill_between(Z, f_growth + f_growth_err, f_growth - f_growth_err, facecolor='blue',alpha=0.9,label=r"$f(z)\pm 1\sigma$")
    plt.fill_between(Z, f_growth + 2.*f_growth_err, f_growth - 2.*f_growth_err, facecolor='lightblue',alpha=0.6,label=r"$f(z) \pm 2\sigma$")
    plt.fill_between(Z, f_growth + 3.*f_growth_err, f_growth - 3.*f_growth_err, facecolor='lightblue',alpha=0.4,label=r"$f(z) \pm 3\sigma$")
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$f(z)$', fontsize=fontsize)
    plt.legend(fontsize='8', loc='upper left')
    
    
    plt.subplot(3,2,5)
    plt.xlim(xmin, xmax)
    plt.ylim(0.20, 2.2)
    #plt.xscale('log')
    plt.fill_between(Z, recomz + sigma_omz, recomz - sigma_omz, facecolor='blue',alpha=0.9)
    plt.fill_between(Z, recomz + 2*sigma_omz, recomz - 2*sigma_omz, facecolor='lightblue',alpha=0.6)
    plt.fill_between(Z, recomz + 3*sigma_omz, recomz - 3*sigma_omz, facecolor='lightblue',alpha=0.4)
    plt.plot(Z, recomz,linewidth=1.5, color='blue')
    plt.plot(Z, Om_SH0ES*(1.+Z)**3/(h_lcdm/H0_SH0ES)**2, linestyle=(5, (10, 3)), linewidth=1.5, color='black')
    plt.vlines(zomone,ymin=0.20,ymax=2.2,colors='k',linestyles='dashed',alpha=0.4)
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$\Omega_{m}(z)$', fontsize=fontsize)
    plt.legend(("Rec $1\sigma$","Rec $2\sigma$","Rec $3\sigma$","Rec",r"$\Omega_m(z)$ $\Lambda$CDM","$z(\Omega_m=1)$"), fontsize='10', loc='upper left')
    
    
    ax = plt.subplot(3,2,6)
    plt.xlim(xmin, xmax)
    plt.ylim(-1, 1.0)
    #plt.ylim(-0.1, 1.1)
    #plt.xscale('log')
    #plt.plot(Z, gammas_org)
    plt.fill_between(Z, gammas + gammas_err, gammas - gammas_err, facecolor='blue',alpha=0.9)
    plt.fill_between(Z, gammas + 2*gammas_err, gammas - 2*gammas_err, facecolor='lightblue',alpha=0.6)
    plt.fill_between(Z, gammas + 3*gammas_err, gammas - 3*gammas_err, facecolor='lightblue',alpha=0.4)
    plt.plot(Z, gammas,linewidth=1.5, color='blue')
    #plt.vlines(zomone,ymin=-4.0,ymax=4.0,colors='k',linestyles='dashed',alpha=0.4)
    plt.hlines(0.42,xmin, xmax, colors='r',linestyles='dashed',alpha=0.5)
    plt.hlines(0.55,xmin, xmax, colors='b',linestyles='dashed',alpha=0.5)
    plt.hlines(0.68,xmin, xmax, colors='g',linestyles='dashed',alpha=0.5)
    #plt.axvspan(1.3615555555555554, 2.0388548548548546, alpha=0.2, color = 'red')
    plt.xlabel('$z$', fontsize=fontsize)
    plt.ylabel('$\gamma(z)$', fontsize=fontsize)
    #plt.legend(("Rec $1\sigma$","Rec $2\sigma$","Rec $3\sigma$","Rec","$z(\Omega_m=1)$","$f(R)$","GR $\Lambda$CDM","DGP"), fontsize='7', loc='lower left')
    plt.legend(("Rec $1\sigma$","Rec $2\sigma$","Rec $3\sigma$","Rec","$f(R)$","GR $\Lambda$CDM","DGP"), fontsize='8', loc='lower left')
    """
    # add figure for subplots
    inset_axes(ax, width="100%", height="100%",
                    bbox_to_anchor=(.05, .08, .35, .35),
                    bbox_transform=ax.transAxes,loc='lower left')
    plt.xlim(xmin, 0.75)
    plt.ylim(0.3, 1.2)
    plt.xticks(np.arange(xmin, 0.75, step=0.2))
    plt.yticks(np.arange(0.3, 1.0, step=0.2))
    plt.fill_between(Z, gammas + gammas_err, gammas - gammas_err, facecolor='blue',alpha=0.2)
    plt.fill_between(Z, gammas + 2*gammas_err, gammas - 2*gammas_err, facecolor='lightblue',alpha=0.4)
    plt.fill_between(Z, gammas + 3*gammas_err, gammas - 3*gammas_err, facecolor='lightblue',alpha=0.6)
    plt.plot(Z, gammas)
    plt.hlines(0.42,xmin, 0.75, colors='r',linestyles='dashed',alpha=0.4)
    plt.hlines(0.55,xmin, 0.75, colors='b',linestyles='dashed',alpha=0.4)
    plt.hlines(0.68,xmin, 0.75, colors='g',linestyles='dashed',alpha=0.4)
    """
    plt.savefig(filename+'plot_DC_fs8.pdf')
    
if __name__=="__main__":
    #filename = 'Growth_Index_SN_fsigma8'
    #filename = 'Growth_Index_SN_fsigma8_SquaredExponential'
    filename = 'Growth_Index_SN_fsigma8_SquaredExponential_DLOBS'
    #filename = 'Growth_Index_SN_fsigma8_Matern92'
    (X,Y,Sigma) = loadtxt("f_rec_data_"+filename+".txt", unpack='True')
    (dX,dY,dSigma) = loadtxt("df_rec_data_"+filename+".txt", unpack='True')
    rec = loadtxt(filename+"_fdc.txt")
    drec = loadtxt(filename+"_dfdc.txt")
    d2rec = loadtxt(filename+"_d2fdc.txt")
    fs8rec = loadtxt(filename+"_fs8.txt")
    dfs8rec = loadtxt(filename+"_dfs8.txt")
    fs8obs = loadtxt(filename+"_fsigma8obs.txt")
    
    plot(X, Y, Sigma, dX, dY, dSigma, rec, drec, d2rec, fs8rec, dfs8rec, fs8obs, xmin=None,xmax=None, filename=filename)
