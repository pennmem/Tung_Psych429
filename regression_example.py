
### Leave-one-session out cross validation
from sklearn.externals import joblib
from sklearn import linear_model
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from math import sqrt
import numpy as np
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn import svm # svm
#import xgboost as xgb  # xgboost
#import deepdish as dd
import os
import shutil
import scipy.io as sio
import sys
import json
import matplotlib.pyplot as plt
import re

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

import scipy

import collections

import sklearn
import os.path as op
import nibabel as nib
from numpy.random import permutation
from matplotlib import cm
import matplotlib as mpl
from nibabel.freesurfer import read_label
from surfer import Brain
from surfer.io import read_stc


import seaborn as sns

from sklearn.externals import joblib

mount_point = '/Volumes/RHINO'  # rhino mount point

# read in data (warning: some subjects might not have all of the data but most should)
# i have some precomputed data saved in these directories but feel free to use your own data
subjects = np.sort(os.listdir(mount_point+ '/scratch/tphan/frequency_selection/'))  # get all subjects

subject = "R1065J"


subject_dir = mount_point + '/scratch/tphan/frequency_selection/' + subject + '/'
# subject_dir_long = mount_point +  '/scratch/tphan/CatFR1_reports_long/' + subject + '/'
dataset_dir = subject_dir + subject + '-dataset_current_freqs.pkl'
bipolar_dir = subject_dir + subject + '-bp_tal_structs.pkl'


dataset = joblib.load(dataset_dir)

y = dataset['recalled']
event_sessions = dataset['sess']
n_sess = len(np.unique(event_sessions))
list_sessions = dataset['list']
bp_tal_structs = dataset['bp_tal_structs']
N_frequency = 8

serialpos = dataset['serialpos']
pow_mat = dataset['pow-mat']
#pow_mat = normalize_sessions(pow_mat, event_sessions)
total_elec = int(pow_mat.shape[1]/8)


frequencies = np.logspace(np.log10(3), np.log10(180), 8)



pow_mat_elec = pow_mat[:,:8]
pow_mat_elec_vec = pow_mat_elec.T.reshape((1800*8,1))

data = pd.DataFrame(pow_mat_elec_vec, columns = ['log_pow'])
data['log_frequency'] = np.repeat(np.log10(frequencies),1800)





import seaborn as sns

fig = plt.figure(figsize = (7,5))
ax = plt.subplot(111)
sns.regplot('log_frequency','log_pow', data = data, label = 'obs', ax = ax, ci = 95, scatter_kws = {'s':20}, line_kws = {'lw':4})


from sklearn import linear_model
reg = linear_model.LinearRegression()

X = data['log_frequency'].values.reshape((len(data['log_pow']),1))
y = data['log_pow'].values

model = reg.fit(X, y)
b_est = model.coef_

sns.set(style = 'whitegrid')
sns.residplot('log_frequency', 'log_pow', data = data, label = 'residual', ax = ax, scatter_kws = {'s':20}, color = 'red')
ax.legend(fontsize =14)
ax.axhline(y = 0, color = 'red', lw = 4, linestyle = 'dashed')

ax.tick_params(labelsize = 13)
ax.set_xlabel('Log Frequency', fontsize = 13)
ax.set_ylabel('Log Power', fontsize = 13)
ax.set_title('R1065J')

fig.savefig('pow_vs_frequency.pdf', dpi = 500)


from statsmodels.formula.api import ols
model = ols("log_pow ~ log_frequency", data).fit()
print(model.summary())
b_est = model.coef_

# F test
model.f_test("log_frequency =0")
R = [[0 , 1]]
model.f_test(R)


# observations bootstrap

N = data.shape[0]
B = 1000
b_boot_obs = np.zeros(B)
b_boot_resid = np.zeros(B)


# residual sampling
y_hat = model.predict(X)
resid = y - y_hat

for b in np.arange(B):
    indices_b = np.random.choice(range(0, N),N)
    y_b = y[indices_b]
    X_b = X[indices_b,:]
    model = reg.fit(X_b,y_b)
    b_boot_obs[b] = model.coef_


    resid_b = resid[indices_b]
    y_b = y_hat + resid_b
    model = reg.fit(X, y_b)
    b_boot_resid[b] = model.coef_




fig = plt.figure(figsize = (7,5))
ax = plt.subplot(111)
sns.distplot(b_boot_obs, label = 'observation resampling', ax = ax, color = 'yellow')
sns.distplot(b_boot_resid, label = 'residual resampling', ax = ax, color = 'grey')
ax.axvline(x = b_est, color = 'red', lw = 4)

ax.tick_params(labelsize = 13)
ax.set_xlabel('bootstrapped values', fontsize = 13)
ax.set_ylabel('', fontsize = 13)
ax.set_title('Bootstrap Sampling Distributions', fontsize = 13)
ax.legend(fontsize =13)
fig.savefig('bootstrap.pdf', dpi = 500)

sd_boot_obs = np.std(b_boot_obs)
sd_boot_resid = np.std(b_boot_resid)
