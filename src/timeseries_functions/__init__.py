import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
import plotly.graph_objects as go
import plotly.io as plio
import datetime
import warnings
import logging
import scipy.stats as stats
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats.distributions import chi2 
from math import sqrt

logging.basicConfig(level=logging.INFO)

def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2-L1))    
    p = chi2.sf(LR, DF).round(3)
    return p

# create a N-order differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    history = list(history)
    n = len(yhat)
    if len(yhat) == 1:
        value = yhat[i] + history[-interval]
        history.append(value)
    else:
        value = yhat[0] + history[-interval]
        history.append(value)
        
        for i in range(1, n):
            value = yhat[i] + history[-interval]
            history.append(value)
    return np.array(history[-n:])

def plot_serie(df_test, predicitions_differenced_5_0_5):

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
    
    return fig