import numpy as np
import pandas as pd

import os
import sys

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


sns.set_style('whitegrid')

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return fig

# Noise
np.random.seed(1)

# plot of discrete white noise
randser = np.random.normal(size=1000)
tsplot(randser, lags=30)


# plot white noise
randser = pd.Series(randser)
fig = plt.figure(figsize = (8,6))
ts_ax = plt.subplot(211)
qq_ax = plt.subplot(212)
randser.plot(ax = ts_ax)
sm.qqplot(randser, line = 's', ax = qq_ax)
qq_ax.set_title('QQ Plot')
ts_ax.set_title('White Noise')
ts_ax.set_xlabel('', size = 15)

fig.savefig('white_noise.pdf', dpi = 500)

# AR models
np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a * x[t - 1] + w[t]

plot_ar = tsplot(x, lags=30)
plot_ar.suptitle('AR Model')
plot_ar.savefig('ar_model_simulated.pdf', dpi = 500)