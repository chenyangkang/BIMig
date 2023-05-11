
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import arviz as az
import pymc as pm
import pickle
import seaborn as sns
from tqdm import tqdm
import time
from warnings import filterwarnings
import contextily as cx
import pickle
import pytensor as pt
filterwarnings('ignore')
pd.set_option('display.max_row', 100)


import pymc as pm
from pytensor.tensor import TensorVariable
import scipy
import jax
import tensorflow_probability.substrates.jax as tfp
jax.scipy.special.erfcx = tfp.math.erfcx

import pymc.sampling.jax as pmjax
from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()


class Site_Migration_Dist:

    ######## logpdf functions
    @staticmethod
    def fc_dist(First_come_mu, First_come_std, nu):
        norm = pm.StudentT('First_come',mu=First_come_mu, sigma=First_come_std, nu=nu)
        return norm
    
    @staticmethod
    def fg_dist(First_gone_mu, First_gone_std, nu):
        norm = pm.StudentT('First_gone',mu=First_gone_mu, sigma=First_gone_std, nu=nu)
        return norm
    
    @staticmethod
    def sc_dist(Second_come_mu, Second_come_std, nu):
            norm = pm.StudentT('Second_come',mu=Second_come_mu, sigma=Second_come_std, nu=nu)
            return norm
    
    @staticmethod
    def sg_dist(Second_gone_mu, Second_gone_std, nu):
            norm = pm.StudentT('Second_gone',mu=Second_gone_mu, sigma=Second_gone_std, nu=nu)
            return norm
        
        
    ###########
    @classmethod
    def _logpdf(cls, x, First_come_mu: TensorVariable, 
                First_come_std: TensorVariable,First_come_nu: TensorVariable, First_come_scaling_factor: TensorVariable,
                        First_gone_mu: TensorVariable, First_gone_std: TensorVariable,First_gone_nu: TensorVariable, First_gone_scaling_factor: TensorVariable,
                    Second_come_mu: TensorVariable, Second_come_std: TensorVariable, Second_come_nu: TensorVariable, Second_come_scaling_factor: TensorVariable,
                    Second_gone_mu: TensorVariable, Second_gone_std: TensorVariable, Second_gone_nu: TensorVariable, Second_gone_scaling_factor: TensorVariable):
        
        combiend_logpdf = pm.math.exp(pm.logp(cls.fc_dist(First_come_mu, First_come_std, First_come_nu), x)) * First_come_scaling_factor+ \
                     pm.math.exp(pm.logp(cls.fg_dist(First_gone_mu, First_gone_std, First_gone_nu), x)) * -First_gone_scaling_factor + \
                     pm.math.exp(pm.logp(cls.sc_dist(Second_come_mu, Second_come_std, Second_come_nu), x)) * Second_come_scaling_factor + \
                     pm.math.exp(pm.logp(cls.sg_dist(Second_gone_mu, Second_gone_std, Second_gone_nu), x))* -Second_gone_scaling_factor

        combiend_logpdf = pm.math.log(combiend_logpdf)
        
        return combiend_logpdf
    
    @classmethod
    def _logcdf(cls, x, First_come_mu: TensorVariable, 
                First_come_std: TensorVariable,First_come_nu: TensorVariable, First_come_scaling_factor: TensorVariable,
                        First_gone_mu: TensorVariable, First_gone_std: TensorVariable,First_gone_nu: TensorVariable, First_gone_scaling_factor: TensorVariable,
                    Second_come_mu: TensorVariable, Second_come_std: TensorVariable, Second_come_nu: TensorVariable, Second_come_scaling_factor: TensorVariable,
                    Second_gone_mu: TensorVariable, Second_gone_std: TensorVariable, Second_gone_nu: TensorVariable, Second_gone_scaling_factor: TensorVariable):
        
        combiend_logcdf =  pm.math.dot(pm.math.exp(pm.logcdf(cls.fc_dist(First_come_mu, First_come_std, First_come_nu), x)), First_come_scaling_factor)+ \
                     pm.math.dot(pm.math.exp(pm.logcdf(cls.fg_dist(First_gone_mu, First_gone_std, First_gone_nu), x)), -First_gone_scaling_factor)+ \
                     pm.math.dot(pm.math.exp(pm.logcdf(cls.sc_dist(Second_come_mu, Second_come_std, Second_come_nu), x)), Second_come_scaling_factor) + \
                     pm.math.dot(pm.math.exp(pm.logcdf(cls.sg_dist(Second_gone_mu, Second_gone_std, Second_gone_nu), x)), -Second_gone_scaling_factor)
                     
        combiend_logcdf = pm.math.log(combiend_logcdf)
        return combiend_logcdf
    
    @classmethod
    def _random(cls, First_come_mu, First_come_std, First_come_scaling_factor,
                First_gone_mu, First_gone_std, First_gone_scaling_factor,
                Second_come_mu, Second_come_std, Second_come_scaling_factor,
                Second_gone_mu, Second_gone_std, Second_gone_scaling_factor, 
                rng=None, size=None):
        
        # Define the distribution objects
        fc_dist = cls.fc_dist(First_come_mu, First_come_std)
        fg_dist = cls.fg_dist(First_gone_mu, First_gone_std)
        sc_dist = cls.sc_dist(Second_come_mu, Second_come_std)
        sg_dist = cls.sg_dist(Second_gone_mu, Second_gone_std)
        
        # Generate random samples from each distribution
        fc_samples = fc_dist.random(rng=rng, size=size) * First_come_scaling_factor
        fg_samples = fg_dist.random(rng=rng, size=size) * First_gone_scaling_factor
        sc_samples = sc_dist.random(rng=rng, size=size) * Second_come_scaling_factor
        sg_samples = sg_dist.random(rng=rng, size=size) * Second_gone_scaling_factor
        
        # Combine the samples into a single tensor
        samples = fc_samples + fg_samples + sc_samples + sg_samples
        
        return samples
    

def assign_spatial_temporal_grid(sub):
    '''
        Spatial-temporal-urban-cropland subsampling. Five dimention. Booooom!
    '''
    lon_array = np.arange(sub.longitude.min(), sub.longitude.max(), 0.05)
    lat_array =  np.arange(sub.latitude.min(), sub.latitude.max(), 0.05)
    doy_array =  np.arange(sub.DOY.min(), sub.DOY.max(), 1)
    sub['lon_lat_doy_urban_crop_grid'] = [str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'_'+str(m) for i,j,k,l,m in zip(np.digitize(sub.longitude, lon_array),
                            np.digitize(sub.latitude, lat_array),
                            np.digitize(sub.DOY, doy_array),
                            np.where(sub.cropland>0,1,0),
                               np.where(sub.urban_areas>0,1,0)
                               )]
    
    return sub

class BIMig():
    def __init__(self):
        pass
    
    def Make_model(self, data):
        with pm.Model() as model:
            
            ### data
            occ_obs = data.occ.values
            DOY_obs = data.DOY.values
            
            ### mu & sigma
            first_come_mu = pm.TruncatedNormal('first_come_mu', mu=40, sigma=100, lower=-90,upper=200)
            first_come_sigma = pm.TruncatedNormal('first_come_sigma', mu=0, sigma=100, upper=60, lower=0)
            # first_come_nu = pm.TruncatedNormal('first_come_nu', mu=1, sigma=100, upper=100, lower=0)
            first_come_nu = 3
            
            first_come_to_first_gone = pm.HalfNormal('first_come_to_first_gone', sigma=100)
            first_gone_to_second_come = pm.HalfNormal('first_gone_to_second_come', sigma=100)
            second_come_to_second_gone = pm.HalfNormal('second_come_to_second_gone', sigma=100)
            
            first_gone_mu = pm.Deterministic('first_gone_mu', first_come_mu + first_come_to_first_gone)
            first_gone_sigma = pm.TruncatedNormal('first_gone_sigma', mu=0, sigma=100, upper=60, lower=0)
            # first_gone_nu = pm.TruncatedNormal('first_gone_nu', mu=1, sigma=100, upper=100, lower=0)
            first_gone_nu  =3
            
            second_come_mu = pm.Deterministic('second_come_mu', first_gone_mu + first_gone_to_second_come)
            second_come_sigma = pm.TruncatedNormal('second_come_sigma', mu=0, sigma=100, upper=60, lower=0)
            # second_come_nu = pm.TruncatedNormal('second_come_nu', mu=1, sigma=100, upper=100, lower=0)
            second_come_nu = 3
            
            second_gone_mu = pm.Deterministic('second_gone_mu', second_come_mu + second_come_to_second_gone)
            second_gone_sigma = pm.TruncatedNormal('second_gone_sigma', mu=0, sigma=100, upper=60, lower=0)
            # second_gone_nu = pm.TruncatedNormal('second_gone_nu', mu=1, sigma=100, upper=100, lower=0)
            second_gone_nu = 3
            
            #### scaler
            first_come_scaler = pm.TruncatedNormal('first_come_scaler', mu=0, lower=0, sigma=1, upper=1)
            first_gone_scaler = pm.TruncatedNormal('first_gone_scaler', mu=0, lower=0, sigma=1, upper=1)
            second_come_scaler = pm.TruncatedNormal('second_come_scaler', mu=0, lower=0, sigma=1, upper=1)
            second_gone_scaler = pm.TruncatedNormal('second_gone_scaler', mu=0, lower=0, sigma=1, upper=1)
            scaler_sum = pm.math.sum(
                pm.math.stack([
                    first_come_scaler,
                    first_gone_scaler,
                    second_come_scaler,
                    second_gone_scaler
                ])
            )
            
            first_come_scaler = first_come_scaler/scaler_sum
            first_gone_scaler = first_gone_scaler/scaler_sum
            second_come_scaler = second_come_scaler/scaler_sum
            second_gone_scaler = second_gone_scaler/scaler_sum
            
            cdf = Site_Migration_Dist._logcdf(
                                            DOY_obs,
                                            first_come_mu, ### first come mean
                                            first_come_sigma, ### first come std
                                            first_come_nu,
                                            first_come_scaler, ### first come scaling factor
                                            
                                            first_gone_mu, ### first gone mean
                                            first_gone_sigma,  ### first gone std
                                            first_gone_nu,
                                            first_gone_scaler, ### first gone scaling factor
                                            
                                            second_come_mu,  ### second come mean
                                            second_come_sigma,  ### second come std
                                            second_come_nu,
                                            second_come_scaler, ### second come scaling factor
                                            
                                            second_gone_mu, ### second gone mean
                                            second_gone_sigma, ### second gone mean
                                            second_gone_nu,
                                            second_gone_scaler, ### second gone mean
                                            )

            # cdf = pm.logcdf(dist, DOY_obs)
            alpha = pm.Normal('alpha', mu=0, sigma=100)
            beta = pm.Normal('beta', mu=0, sigma=100)
            p = pm.Deterministic('p',pm.math.sigmoid(cdf * alpha + beta))
            y_ = pm.Bernoulli('y_', p=p, observed = occ_obs)
            # obs_prob = pm.Deterministic('obs_prob',cdf)
        # self.model = model
        return model


    def sample_model(model, samples=1000, tune=1000, chains=2, cores=2):
        with model:
            idata = pmjax.sample_numpyro_nuts(samples,tune=tune, chains=chains, cores=cores)
            # 
        return idata


    def eval_cdf_func(
        self,
        DOY_obs,
        first_come_mu, ### first come mean
        first_come_sigma, ### first come std
        first_come_scaler, ### first come scaling factor
        
        first_gone_mu, ### first gone mean
        first_gone_sigma,  ### first gone std
        first_gone_scaler, ### first gone scaling factor
        
        second_come_mu,  ### second come mean
        second_come_sigma,  ### second come std
        second_come_scaler, ### second come scaling factor
        
        second_gone_mu, ### second gone mean
        second_gone_sigma, ### second gone mean
        second_gone_scaler, ### second gone mean
    ):
        cdf1 = pm.logcdf(
                pm.Normal.dist(mu=first_come_mu, sigma=first_come_sigma),DOY_obs
            ).eval()
        cdf2 = pm.logcdf(
                pm.Normal.dist(mu=first_gone_mu, sigma=first_gone_sigma),DOY_obs
            ).eval()
        cdf3 = pm.logcdf(
                pm.Normal.dist(mu=second_come_mu, sigma=second_come_sigma),DOY_obs
            ).eval()
        cdf4 = pm.logcdf(
                pm.Normal.dist(mu=second_gone_mu, sigma=second_gone_sigma),DOY_obs
            ).eval()
        
        logcdf = np.log(np.exp(cdf1) * first_come_scaler + \
                    np.exp(cdf2) * -first_gone_scaler +\
                        np.exp(cdf3) * second_come_scaler + \
                            np.exp(cdf4) * -second_gone_scaler)
        return logcdf
        
        
        

    def predict(self, idata, data):

        ### data
        # occ_obs_ = data.occ.values
        DOY_obs_ = data.DOY.values
        
        ### mu & sigma
        first_come_mu_ = np.concatenate(idata.posterior['first_come_mu'], axis=0).mean(axis=0)
        first_come_sigma_ = np.concatenate(idata.posterior['first_come_sigma'], axis=0).mean(axis=0)
        
        first_come_to_first_gone_ = np.concatenate(idata.posterior['first_come_to_first_gone'], axis=0).mean(axis=0)
        first_gone_to_second_come_ = np.concatenate(idata.posterior['first_gone_to_second_come'], axis=0).mean(axis=0)
        second_come_to_second_gone_ = np.concatenate(idata.posterior['second_come_to_second_gone'], axis=0).mean(axis=0)
        
        first_gone_mu_ = first_come_mu_ + first_come_to_first_gone_
        first_gone_sigma_ = np.concatenate(idata.posterior['first_gone_sigma'], axis=0).mean(axis=0)
        
        second_come_mu_ = first_gone_mu_ + first_gone_to_second_come_
        second_come_sigma_ = np.concatenate(idata.posterior['second_come_sigma'], axis=0).mean(axis=0)
        
        second_gone_mu_ = second_come_mu_ + second_come_to_second_gone_
        second_gone_sigma_ = np.concatenate(idata.posterior['second_gone_sigma'], axis=0).mean(axis=0)
        
        #### scaler
        first_come_scaler_ = np.concatenate(idata.posterior['first_come_scaler'], axis=0).mean(axis=0)
        first_gone_scaler_ = np.concatenate(idata.posterior['first_gone_scaler'], axis=0).mean(axis=0)
        second_come_scaler_ = np.concatenate(idata.posterior['second_come_scaler'], axis=0).mean(axis=0)
        second_gone_scaler_ = np.concatenate(idata.posterior['second_gone_scaler'], axis=0).mean(axis=0)
        
        scaler_sum_ = np.sum(
            np.stack([
                first_come_scaler_,
                first_gone_scaler_,
                second_come_scaler_,
                second_gone_scaler_
            ])
        )
        
        first_come_scaler_ = first_come_scaler_/scaler_sum_
        first_gone_scaler_ = first_gone_scaler_/scaler_sum_
        second_come_scaler_ = second_come_scaler_/scaler_sum_
        second_gone_scaler_ = second_gone_scaler_/scaler_sum_
        
        
        cdf_ = self.eval_cdf_func(
                                DOY_obs_,
                                first_come_mu_, ### first come mean
                                first_come_sigma_, ### first come std
                                first_come_scaler_, ### first come scaling factor
                                
                                first_gone_mu_, ### first gone mean
                                first_gone_sigma_,  ### first gone std
                                first_gone_scaler_, ### first gone scaling factor
                                
                                second_come_mu_,  ### second come mean
                                second_come_sigma_,  ### second come std
                                second_come_scaler_, ### second come scaling factor
                                
                                second_gone_mu_, ### second gone mean
                                second_gone_sigma_, ### second gone mean
                                second_gone_scaler_, ### second gone mean
                                )

        # cdf = pm.logcdf(dist, DOY_obs)
        alpha_ = np.concatenate(idata.posterior['alpha'], axis=0).mean(axis=0)
        beta_ = np.concatenate(idata.posterior['beta'], axis=0).mean(axis=0)
        
        from scipy.special import expit
        p_ = expit(cdf_ * alpha_ + beta_)
        # y__ = np.where(p>0.5,1,0)
        return p_, cdf_

    def plot_prediction(self, p_, data):
        plt.plot(
                pd.DataFrame({
                'DOY':data.DOY.values,
                'predicted_p':p_
            }).groupby('DOY').mean().rolling(10).mean(),
                label = 'Predicted'
        )

        plt.plot(
            data[['DOY','occ']].groupby('DOY').mean().rolling(10).mean(),
            label='Truth'
        )
        plt.legend()
        plt.show()
            
    def plot_midpoint(self, idata):
        
        for var in ['first_come_mu','first_gone_mu','second_come_mu','second_gone_mu']:
            
            plt.hist(
                np.concatenate(idata.posterior[var].values, axis=0),
                label=var.replace('_',' '),bins=100
            )
            
        plt.legend(bbox_to_anchor=(1,1))
        plt.title('Uncertainty of midpoint')
        plt.xlabel('Day of Year')
        plt.show()
        
    def plot_mean_mid_and_sigma(self, idata):
        for var in ['first_come','first_gone','second_come','second_gone']:
            mu = np.concatenate(idata.posterior[var + '_mu'].values, axis=0).mean()
            sigma = np.concatenate(idata.posterior[var + '_sigma'].values, axis=0).mean()
            x = np.arange(mu-3*sigma, mu+3*sigma, 1)
            y = scipy.stats.norm.pdf(x = x, loc = mu, scale = sigma)
            plt.plot(x,y)
        plt.title('Passby using mean mu and sigma')
        plt.xlabel('Day of Year')
        plt.show()     
                
    def plot_sample_mid_and_sigma(self, idata):

        for var in ['first_come','first_gone','second_come','second_gone']:
            sample_size = 50
            mu = np.concatenate(idata.posterior[var + '_mu'].values, axis=0)[:sample_size]
            sigma = np.concatenate(idata.posterior[var + '_sigma'].values, axis=0)[:sample_size]
            
            for mu_,sigma_ in zip(mu, sigma):
                x = np.arange(mu_-1*sigma_, mu_+1*sigma_, 1)
                y = scipy.stats.norm.pdf(x = x, loc = mu_, scale = sigma_)
                plt.plot(x,y)
            
            # plt.hist(
            #     np.concatenate(idata.posterior['first_come_mu'].values, axis=0),
            #     label='first come',bins=100
            # )

        # plt.legend(bbox_to_anchor=(1,1))
        # plt.title('Uncertainty of midpoint')
        plt.xlabel('Day of Year')
        plt.show()