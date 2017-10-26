import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('power_frequency.csv')
data = data.drop(data.columns[0], axis = 1)



r2.0 = function(sig){
  x <- seq(1,10,length.out = 100)        # our predictor
  y <- 2 + 1.2*x + rnorm(100,0,sd = sig) # our response; a function of x plus some random noise
  summary(lm(y ~ x))$r.squared           # print the R-squared value
}


# pitfalls of R
# R^2 can be made small by inceasing sigma^2
def r2(sig):
    x= np.arange(1,10, step = 0.1)
    y = 0.5 + 1.2*x + np.random.normal(0,sig, len(x))

    data = pd.DataFrame({'x':x, 'y':y})
    model = ols('y ~ x',data).fit()
    r2 = model.rsquared
    return r2

sig_vec = np.arange(1,10, step = 1)
r2_vec = [r2(sig) for sig in sig_vec]

plt.plot(sig_vec, r2_vec,  '-o')

plt.xlabel('sig')
plt.ylabel('r2')


# R^2 close to 1 when model is wrong
sig = 0.1
x= np.arange(1,10, step = 0.1)
y = 0.5 + 1.2*pow(x,2) + np.random.normal(0,sig, len(x))
data = pd.DataFrame({'x':x, 'y':y})
model = ols('y ~ x', data).fit()
r2 = model.rsquared
print(r2)

# Good R^2 but bad model
alpha_est, b_est = model.params
plt.plot(x,y, 'o', color = 'indigo')
plt.plot(x, alpha_est + b_est*x, color = 'red', label = 'LS')

# Changing the range of X
sig = 0.9
x1= np.arange(1,10, step = 0.1)
y1 = 0.5 + 1.2*x1 + np.random.normal(0,sig, len(x1))
data = pd.DataFrame({'x1':x1, 'y1':y1})
model = ols('y1 ~ x1', data).fit()
r2_long = model.rsquared


sig = 0.9
x2= np.arange(1,2, step = 0.01)
y2 = 0.5 + 1.2*x2 + np.random.normal(0,sig, len(x2))
data = pd.DataFrame({'x2':x2, 'y2':y2})
model = ols('y2 ~ x2', data).fit()
r2_short = model.rsquared

plt.plot(x1, y1, color = 'green', label = 'r2 = ' + str(round(r2_long,3)))
plt.plot(x2, y2,color = 'red', label = 'r2 = ' + str(round(r2_short,3)))
plt.legend()




from statsmodels.formula.api import ols
model = ols("log_pow ~ log_frequency", data).fit()
# get model parameters
print(model.params)
print(model.summary())







from sklearn import linear_model
reg = linear_model.LinearRegression()
X = data['log_frequency'].values.reshape((len(data['log_pow']),1))
y = data['log_pow'].values
model = reg.fit(X, y)
b_est = model.coef_
alpha_est = model.intercept_


fig = plt.figure(figsize = (7,5))
ax = plt.subplot(111)

ax.scatter(X,y, label = 'observation')
ax.plot(X, alpha_est + b_est*X, color = 'red', label = 'LS')
ax.legend(fontsize =14)

ax.tick_params(labelsize = 13)
ax.set_xlabel('Log Frequency', fontsize = 13)
ax.set_ylabel('Log Power', fontsize = 13)
ax.set_title('R1065J')
ax.text(1.75,8,'y =' + str(round(alpha_est,2)) + str(round(b_est,2)) + '*x' ,size=12)


#fig.savefig('pow_vs_frequency.pdf', dpi = 500)
#sns.set(style = 'whitegrid')
#sns.residplot('log_frequency', 'log_pow', data = data, label = 'residual', ax = ax, scatter_kws = {'s':20}, color = 'red')


from statsmodels.formula.api import ols
model = ols("log_pow ~ log_frequency", data).fit()
# get model parameters
print(model.params)
print(model.summary())

# Get all the good stuffs
dir(model)
mse_model = model.mse_model
mse_resid = model.mse_resid
mse_total = model.mse_total
df_model = model.df_model
df_resid = model.df_resid
df_total = model.nobs -1

# get sum of squares
tss = mse_total*df_total
rss = mse_resid*df_resid
mss = mse_model*df_model
tss == mss + rss

# f_test
F_stat = mse_model/mse_resid

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
ax.hist(b_boot_obs, label = 'observation resampling', facecolor = 'yellow', alpha = 0.75, bins = 50)
ax.hist(b_boot_resid, label = 'residual resampling', facecolor = 'grey', alpha = 0.75, bins = 50)
ax.axvline(x = b_est, color = 'red', lw = 4)

ax.tick_params(labelsize = 13)
ax.set_xlabel('bootstrapped values', fontsize = 13)
ax.set_ylabel('', fontsize = 13)
ax.set_title('Bootstrap Sampling Distributions', fontsize = 13)
ax.legend(fontsize =13)
fig.savefig('bootstrap.pdf', dpi = 500)

sd_boot_obs = np.std(b_boot_obs)
sd_boot_resid = np.std(b_boot_resid)
