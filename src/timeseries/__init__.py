import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2 
from math import sqrt
import plotly.graph_objects as go
import warnings
import scipy.stats as stats
warnings.filterwarnings("ignore")

from timeseries_functions import *

data = pd.read_csv("ETH.csv", parse_dates = ["Fecha"], dayfirst = True)

# Dropping commas
data['Cierre']=data['Cierre'].str.replace(',','')
data['Apertura']=data['Apertura'].str.replace(',','')
data['Máximo']=data['Máximo'].str.replace(',','')
data['Mínimo']=data['Mínimo'].str.replace(',','')

# Delitting 'K' & '%' from Vol. and % var fields, respectively.
data['Vol.'] = data['Vol.'].map(lambda x: x.rstrip('Kk'))
data['% var.'] = data['% var.'].map(lambda x: x.rstrip('%'))

# Convert fields except "Fecha" in float fields
cols = data.columns.drop('Fecha')
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

# Updating Vol. and % var fields with the real value
data['Vol.'] = data['Vol.'].apply(lambda x: x*1000)
data['% var.'] = data['% var.'].apply(lambda x: x/100)

df_comp = data.copy()
df_comp.set_index("Fecha", inplace=True)
df_comp=df_comp.sort_index().asfreq('D')

# Splitting data in training and testing
size = int(len(df_comp)*0.9)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

# Testing stationarity
test_statistic, p_value, used_lag, n_obs, critical_values, maximized_information_criterion = sts.adfuller(df.Cierre)

for k,v in critical_values.items():
    if test_statistic < v:
        print("Data is stationary with a %.2f%% of confidence" % (1-(float(k.rstrip('%')))/100))
    else:
        print("There isn't enough proves to reject that data isn't stationary with a %.2f%% of confidence"% (1-(float(k.rstrip('%')))/100))


x = df.Cierre.values

# Regarding 1 ordered differenced time serie
differenced = difference(x, 1)

# Testing stationarity
test_statistic, p_value, used_lag, n_obs, critical_values, maximized_information_criterion = sts.adfuller(differenced)

for k,v in critical_values.items():
    if test_statistic < v:
        print("Data is stationary with a %.2f%% of confidence" % (1-(float(k.rstrip('%')))/100))
    else:
        print("There isn't enough proves to reject that data isn't stationary with a %.2f%% of confidence"% (1-(float(k.rstrip('%')))/100))

## ARIMA(1,0,1)

model_ar_1_i_0_ma_1 = ARIMA(differenced, order=(1,0,1))
results_ar_1_i_0_ma_1 = model_ar_1_i_0_ma_1.fit()
results_ar_1_i_0_ma_1.summary()

## Residuals of the ARIMA(1,0,1)

df['res_ar_1_i_0_ma_1'] = np.concatenate([0,results_ar_1_i_0_ma_1.resid], axis=None)
sgt.plot_acf(df.res_ar_1_i_0_ma_1[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(1,0,1)",size=20)
plt.show()

print("ARIMA(5,0,5):  \t LL = ", results_ar_5_i_0_ma_5_differenced.llf, "\t AIC = ", results_ar_5_i_0_ma_5_differenced.aic)
print("ARIMA(5,0,4):  \t LL = ", results_ar_5_i_0_ma_4_differenced.llf, "\t AIC = ", results_ar_4_i_0_ma_5_differenced.aic)
print("ARIMA(4,0,5):  \t LL = ", results_ar_4_i_0_ma_5_differenced.llf, "\t AIC = ", results_ar_4_i_0_ma_5_differenced.aic)
print("ARIMA(5,0,3):  \t LL = ", results_ar_5_i_0_ma_3_differenced.llf, "\t AIC = ", results_ar_5_i_0_ma_3_differenced.aic)
print("ARIMA(3,0,5):  \t LL = ", results_ar_3_i_0_ma_5_differenced.llf, "\t AIC = ", results_ar_3_i_0_ma_5_differenced.aic)


model_ar_5_i_0_ma_5_differenced = ARIMA(differenced, order=(5,0,5))
results_ar_5_i_0_ma_5_differenced = model_ar_5_i_0_ma_5_differenced.fit()
results_ar_5_i_0_ma_5_differenced.summary()

model1 = results_ar_5_i_0_ma_4_differenced
model2 = results_ar_5_i_0_ma_5_differenced
DF = 1
if LLR_test(model1, model2, DF =DF) < .05:
    print("LLR test p-value = " + str(LLR_test(model1, model2, DF =DF)))
    print("There is enough evidence that {model2} is statiscally significant than {model1}".format(model2="ARIMA(5,0,5)", model1="ARIMA(5,0,4)"))
else:
    print("LLR test p-value = " + str(LLR_test(model1, model2, DF =DF)))
    print("There isn't enough evidence that {model2} is statiscally significant than {model1}".format(model2="ARIMA(5,0,5)", model1="ARIMA(5,0,4)"))

# look of residuals
df['res_ar_5_i_0_ma_5_differenced'] = np.concatenate(([0], results_ar_5_i_0_ma_5_differenced.resid))
sgt.plot_acf(df.res_ar_5_i_0_ma_5_differenced[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for ARIMA(5,0,5)",size=20)
plt.show()

# Plotting residuals
df.res_ar_5_i_0_ma_5_differenced.plot(figsize = (20,5))
plt.title("Residuals ARMA(5,0,5)", size=24)
plt.show()

# Testing stationarity of residuals
test_statistic, p_value, used_lag, n_obs, critical_values, maximized_information_criterion = sts.adfuller(df.res_ar_5_i_0_ma_5_differenced)

print("Test-statistic \t\t", test_statistic)
print("P-value \t\t", p_value)
print("\n")
for k,v in critical_values.items():
    if test_statistic < v:
        print("Data is stationary with a %.2f%% of confidence" % (1-(float(k.rstrip('%')))/100))
    else:
        print("There isn't enough proves to reject that data isn't stationary with a %.2f%% of confidence"% (1-(float(k.rstrip('%')))/100))

# Let’s test our model prediciting one week values
forecast_differenced_5_0_5 = results_ar_5_i_0_ma_5_differenced.forecast(7)[0] 

# Returning predicitions to the original scale
predicitions_differenced_5_0_5 = inverse_difference(df.Cierre, forecast_differenced_5_0_5, interval = 1)

# importing the necessary functions to calc the RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
from math import sqrt

sqrt(mean_squared_error(df_test['Cierre'][0:7].values, predicitions_differenced_5_0_5))

# Opción 1
fig = plot_serie(df_test, predicitions_differenced_5_0_5)
fig.show()

# Opción 2
aux_df = pd.DataFrame({"Real_value": df_test['Cierre'][0:7].values, \
                       "Forecast": np.round(predicitions_differenced_5_0_5,2)} \
                        , index = df_test['Cierre'][0:7].index)

aux = aux_df.reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(aux['Fecha']),
                y=aux['Real_value'],
                name="Real value",
                marker_color='rgb(55, 83, 109)',
                mode="lines+text",
                text=list(aux['Real_value']),
                textposition="top center"
                ))
fig.add_trace(go.Scatter(x=list(aux['Fecha']),
                y=aux['Forecast'],
                name="Forecast",
                marker_color='rgb(100, 53, 39)',
                mode="lines+text",
                text=list(aux['Forecast']),
                textposition="top center"
                ))
fig.add_trace(go.Scatter(
                name='Upper Bound',
                x=list(aux['Fecha']),
                y=aux['Forecast']+19,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
                ))
fig.add_trace(go.Scatter(
                name='Lower Bound',
                x=list(aux['Fecha']),
                y=aux['Forecast']-19,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
                ))
fig.update_layout(
    title='Predictions vs Actual training with original values',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Price in dollars $',
        titlefont_size=16,
        tickfont_size=14,
    )
)
fig.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX

model_sarimax = SARIMAX(differenced, order=(1,0,1), seasonal_order = (1,0,1,7))
results_sar_1_i_0_ma_1 = model_sarimax.fit()

# Residuals of SARIMAX (1,0,1)X(1,0,1.48)
df['res_sar_1_i_0_ma_1'] = np.concatenate([0, results_sar_1_i_0_ma_1.resid], axis=None)
sgt.plot_acf(df.res_sar_1_i_0_ma_1, zero = False, lags = 40)
plt.title("ACF Of Residuals for SARIMA(1,0,1)x(1,0,1,7)",size=20)
plt.show()

print("SARIMA(5,0,5)x(1,0,1,7):  \t LL = ", results_sar_5_i_0_ma_5.llf, "\t AIC = ", results_sar_5_i_0_ma_5.aic)
print("SARIMA(4,0,5)x(1,0,1,7):  \t LL = ", results_sar_4_i_0_ma_5.llf, "\t AIC = ", results_sar_4_i_0_ma_5.aic)
print("SARIMA(3,0,5)x(1,0,1,7):  \t LL = ", results_sar_3_i_0_ma_5.llf, "\t AIC = ", results_sar_3_i_0_ma_5.aic)
print("SARIMA(5,0,2)x(1,0,1,7):  \t LL = ", results_sar_5_i_0_ma_2.llf, "\t AIC = ", results_sar_5_i_0_ma_2.aic)
print("SARIMA(2,0,5)x(1,0,1,7):  \t LL = ", results_sar_2_i_0_ma_5.llf, "\t AIC = ", results_sar_2_i_0_ma_5.aic)

model_sar_5_i_0_ma_5 = SARIMAX(differenced, order=(5,0,5), seasonal_order = (1,0,1,7))
results_sar_5_i_0_ma_5 = model_sar_5_i_0_ma_5.fit()

results_sar_5_i_0_ma_5.summary()

model1 = results_sar_4_i_0_ma_5
model2 = results_sar_5_i_0_ma_5
DF = 1
if LLR_test(model1, model2, DF =DF) < .05:
    print("LLR test p-value = " + str(LLR_test(model1, model2, DF =DF)))
    print("There is enough evidence that {model2} is statiscally significant than {model1}".format(model2="SARIMA(5,0,5)x(1,0,1,7)", model1="SARIMA(4,0,5)x(1,0,1,7)"))
else:
    print("LLR test p-value = " + str(LLR_test(model1, model2, DF =DF)))
    print("There isn't enough evidence that {model2} is statiscally significant than {model1}".format(model2="SARIMA(5,0,5)x(1,0,1,7)", model1="SARIMA(4,0,5)x(1,0,1,7)"))

# Let’s compare with SARIMA(5,0,5)x(1,0,1,7) with ARIMA(5,0,5).
print("ARIMA(5,0,5):  \t\t\t LL = ", results_ar_5_i_0_ma_5_differenced.llf, "\t AIC = ", results_ar_5_i_0_ma_5_differenced.aic)
print("SARIMA(5,0,5)x(1,0,1,7):  \t LL = ", results_sar_5_i_0_ma_5.llf, "\t AIC = ", results_sar_5_i_0_ma_5.aic)

# Let’s take plot residuals and use ADF test to have enough evidence to say that residuals 
# of the model are a white noise.
# Plotting residuals
df.res_sar_5_i_0_ma_5.plot(figsize = (20,5))
plt.title("Residuals SARIMA(5,0,5)x(1,0,1,7) with differenced data", size=24)
plt.show()

# Testing stationarity of residuals
test_statistic, p_value, used_lag, n_obs, critical_values, maximized_information_criterion = sts.adfuller(df.res_sar_5_i_0_ma_5)

print("Test-statistic \t\t", test_statistic)
print("P-value \t\t", p_value)
print("\n")
for k,v in critical_values.items():
    if test_statistic < v:
        print("Data is stationary with a %.2f%% of confidence" % (1-(float(k.rstrip('%')))/100))
    else:
        print("There isn't enough proves to reject that data isn't stationary with a %.2f%% of confidence"% (1-(float(k.rstrip('%')))/100))

# We’re are getting good results so we’ll proceed predicting and analyzing results.
forecast_sarima_5_0_5 = results_sar_5_i_0_ma_5.forecast(7) 

predicitions_sarima_5_0_5 = inverse_difference(df.Cierre, forecast_differenced_5_0_5, interval = 1)

sqrt(mean_squared_error(df_test['Cierre'][0:7].values, predicitions_sarima_5_0_5))

# Plot
fig = plot_serie(df_test, predicitions_sarima_5_0_5)
fig.show()


model_sar_5_i_0_ma_5_ar2_i0_ma1_m7 = SARIMAX(differenced, order=(5,0,5), seasonal_order = (2,0,1,7))
results_sar_5_i_0_ma_5_ar2_i0_ma1_m7 = model_sar_5_i_0_ma_5_ar2_i0_ma1_m7.fit()
results_sar_5_i_0_ma_5_ar2_i0_ma1_m7.summary()

df['res_sar_5_i_0_ma_5_ar2_i0_ma1_m7'] = np.concatenate(([0], results_sar_5_i_0_ma_5_ar2_i0_ma1_m7.resid))
sgt.plot_acf(df.res_sar_5_i_0_ma_5_ar2_i0_ma1_m7[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for SARIMA(5,0,5)x(2, 0, [1], 7)",size=20)
plt.show()

print("SARIMA(5,0,5)x(1,0,1,7):  \t LL = ", results_sar_5_i_0_ma_5.llf, "\t AIC = ", results_sar_5_i_0_ma_5.aic)
print("SARIMA(5,0,5)x(2,0,1,7):  \t LL = ", results_sar_5_i_0_ma_5_ar2_i0_ma1_m7.llf, "\t AIC = ", results_sar_5_i_0_ma_5_ar2_i0_ma1_m7.aic)


model1 = results_sar_5_i_0_ma_5
model2 = results_sar_5_i_0_ma_5_ar2_i0_ma1_m7
DF = 1
if LLR_test(model1, model2, DF =DF) < .05:
    print("LLR test p-value = " + str(LLR_test(model1, model2, DF =DF)))
    print("There is enough evidence that {model2} is statiscally significant than {model1}".format(model2="SARIMA(5,0,5)x(2,0,1,7)", model1="SARIMA(5,0,5)x(1,0,1,7)"))
else:
    print("LLR test p-value = " + str(LLR_test(model1, model2, DF =DF)))
    print("There isn't enough evidence that {model2} is statiscally significant than {model1}".format(model2="SARIMA(5,0,5)x(2,0,1,7)", model1="SARIMA(5,0,5)x(1,0,1,7)"))


forecast_sarima_5_0_5_ar2_i0_ma1_m7 = results_sar_5_i_0_ma_5_ar2_i0_ma1_m7.forecast(7) 

predicitions_sarima_5_0_5_ar2_i0_ma1_m7 = inverse_difference(df.Cierre, forecast_sarima_5_0_5_ar2_i0_ma1_m7, interval = 1)

sqrt(mean_squared_error(df_test['Cierre'][0:7].values, predicitions_sarima_5_0_5_ar2_i0_ma1_m7))

# Plot
fig = plot_serie(df_test, predicitions_sarima_5_0_5_ar2_i0_ma1_m7)
fig.show()



####################################### Time window 2 #######################################
size2 = int(len(df_comp)*0.95)
df2, df_test2 = df_comp.iloc[:size2], df_comp.iloc[size2:]
# Testing stationarity
test_statistic, p_value, used_lag, n_obs, critical_values, maximized_information_criterion = sts.adfuller(df2.Cierre)

for k,v in critical_values.items():
    if test_statistic < v:
        print("Data is stationary with a %.2f%% of confidence" % (1-(float(k.rstrip('%')))/100))
    else:
        print("There isn't enough proves to reject that data isn't stationary with a %.2f%% of confidence"% (1-(float(k.rstrip('%')))/100))

x2 = df2.Cierre.values

# Regarding 1 ordered differenced time serie
differenced2 = difference(x2, 1)

# Testing stationarity
test_statistic, p_value, used_lag, n_obs, critical_values, maximized_information_criterion = sts.adfuller(differenced2)

for k,v in critical_values.items():
    if test_statistic < v:
        print("Data is stationary with a %.2f%% of confidence" % (1-(float(k.rstrip('%')))/100))
    else:
        print("There isn't enough proves to reject that data isn't stationary with a %.2f%% of confidence"% (1-(float(k.rstrip('%')))/100))

print(len(df2.Cierre))
print(len(differenced2))

model_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2 = SARIMAX(differenced2, order=(5,0,5), seasonal_order = (2,0,1,7))
results_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2 = model_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2.fit()

results_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2.summary()

df2['res_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2'] = np.concatenate(([0], results_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2.resid))
sgt.plot_acf(df2.res_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2[1:], zero = False, lags = 40)
plt.title("ACF Of Residuals for SARIMA(5,0,5)x(2, 0, [1], 7)",size=20)
plt.show()

forecast_sarima_5_0_5_ar2_i0_ma1_m7_window2 = results_sar_5_i_0_ma_5_ar2_i0_ma1_m7_window2.forecast(7) 
forecast_sarima_5_0_5_ar2_i0_ma1_m7_window2

predicitions_sarima_5_0_5_ar2_i0_ma1_m7_window2 = inverse_difference(df2.Cierre, forecast_sarima_5_0_5_ar2_i0_ma1_m7_window2, interval = 1)
predicitions_sarima_5_0_5_ar2_i0_ma1_m7_window2

sqrt(mean_squared_error(df_test2['Cierre'][0:7].values, predicitions_sarima_5_0_5_ar2_i0_ma1_m7_window2))

fig2 = plot_serie(df_test2, predicitions_sarima_5_0_5_ar2_i0_ma1_m7_window2)
fig2.show

# This are the steps to follow for the other time windows please if you want to se code review ETH.ipynb

#######################################################################################################

# Looking for a confidence interval of 𝜇 for the errors based on the sample of the 
# different time windows.
vals = df_predictions['error']
n = len(vals)
x_overline = vals.mean()
sd_aux = []
for i in df_predictions['error']:
    sd_aux.append((i-x_overline)**2)
sd = sqrt(sum(sd_aux)/(n-1))

alpha = 0.05
z_critical = stats.norm.ppf(q = 1-alpha/2)

# Confidence interval
ci = (x_overline-z_critical*(sd/sqrt(n)), x_overline+z_critical*(sd/sqrt(n)))
ci

# Following the same logic of ARIMA & SARIMA models the best GARCH I obtained was this:
model_garch_1_2 = arch_model(differenced, mean = "Constant",  vol = "GARCH", p = 1, q = 2)
results_garch_1_2 = model_garch_1_2.fit(update_freq = 5)
results_garch_1_2.summary()

yhat = results_garch_1_2.forecast(horizon=7)

yhat = yhat.variance.values[-1]

# Since we want volatility and not the variance
result = []
for x in yhat:
    a = sqrt(x)
    result.append(a)
    
result