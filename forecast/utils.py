import numpy as np
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from progressbar import *
from scipy.stats import norm

## 신뢰구간 구축
def confidence_interval(sample, alpha = 0.05):
    mean = np.mean(sample, axis=0)
    std_error = np.std(sample, axis=0)

    max_interval = mean + norm.ppf(alpha/2, loc = 0, scale = 1) * std_error/np.sqrt(len(sample))
    min_interval = mean - norm.ppf(alpha/2, loc = 0, scale = 1) * std_error/np.sqrt(len(sample))
    
    return mean, min_interval, max_interval


def plot_prediction(true, y_pred, trend, seasonality, data):
    fig = make_subplots(
    subplot_titles=['True Vs Predicted','Trend','Seasonality'],
    rows=2, cols=2,
    vertical_spacing=0.15,
    horizontal_spacing=0.04,
    column_widths=[0.9, 0.6],
    row_heights=[0.8, 0.8],
    specs=[[{"rowspan": 2}, {}], [None, {}]])
            
    # confidence interval
    pred_mean, pred_min_interval, pred_max_interval = confidence_interval(y_pred, 0.01)
    trend_mean, trend_min_interval, trend_max_interval = confidence_interval(trend, 0.01)
    seasonal_mean, seasonal_min_interval, seasonal_max_interval = confidence_interval(seasonality, 0.01)
            
    # plot(1,1) - prediction vs real_value
    ## CI
    fig.add_trace(go.Scatter(name = 'Lower Bound',
                             x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp(),
                             y = np.concatenate([true.flatten()[143].reshape(-1), pred_max_interval]),
                             mode = 'lines',
                             marker = dict(color="rgb(179,226,205)"),
                             line = dict(width=0),
                             showlegend = False), row=1,col=1)
    fig.add_trace(go.Scatter(name = 'Confidence Interval',
                             x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp(),
                             y = np.concatenate([true.flatten()[143].reshape(-1), pred_min_interval]),
                             marker = dict(color="rgb(179,226,205)"),
                             line = dict(width=0),
                             mode = 'lines',
                             fillcolor = 'rgba(179,226,205,0.6)',
                             fill = 'tonexty',
                             showlegend = True), row=1,col=1)
    ## mean value
    fig.add_trace(go.Scatter(x = pd.period_range(start = data['Close'].index[0], end = None, periods = 144, freq = '5min').to_timestamp(),
                             y = true.flatten(), 
                             name = "Real value", line=dict(color="#636EFA")), row=1,col=1)
    fig.add_trace(go.Scatter(x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp(), 
                             y = np.concatenate([true.flatten()[143].reshape(-1), pred_mean]), 
                             name = "Prediction average", line=dict(color="red")), row=1,col=1)
            
    # plot(1,2) - trend
    ## CI
    fig.add_trace(go.Scatter(name = 'Lower Bound',
                             x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp()[1:],
                             y = trend_max_interval,
                            mode = 'lines',
                            marker = dict(color="rgb(179,226,205)"),
                            line = dict(width=0),
                            showlegend = False), row=1,col=2)
    fig.add_trace(go.Scatter(name = 'Upper Bound',
                             x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp()[1:],
                             y = trend_min_interval,
                             marker = dict(color="rgb(179,226,205)"),
                             line = dict(width=0),
                             mode = 'lines',
                             fillcolor = 'rgba(179,226,205,0.6)',
                             fill = 'tonexty',
                             showlegend = False), row=1,col=2)
    ## mean value
    fig.add_trace(go.Scatter(x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp()[1:], 
                             y = trend_mean, name = "Trend average", line=dict(color="red"), showlegend = False), row=1,col=2)
            
    # plot(2,2) - seasonality
    ## CI
    fig.add_trace(go.Scatter(name = 'Lower Bound',
                             x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp()[1:],
                             y = seasonal_max_interval,
                             mode = 'lines',
                             marker = dict(color="rgb(179,226,205)"),
                             line = dict(width=0),
                             showlegend = False), row=2,col=2)
    fig.add_trace(go.Scatter(name = 'Upper Bound',
                             x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp()[1:],
                             y = seasonal_min_interval,
                             marker = dict(color="rgb(179,226,205)"),
                             line = dict(width=0),
                             mode = 'lines',
                             fillcolor = 'rgba(179,226,205,0.6)',
                             fill = 'tonexty',
                             showlegend = False), row=2,col=2)
    ## mean value
    fig.add_trace(go.Scatter(x = pd.period_range(start = data['Close'].index[-1], end = None, periods = 49, freq = '5min').to_timestamp()[1:], 
                             y = seasonal_mean, name = "Seasonality average", line=dict(color="red"), showlegend = False), row=2,col=2)
    # dash line
    full_fig = fig.full_figure_for_development()
    fig.add_shape(type="line", xref='x', yref='paper',
                  x0=list(data['Close'].index)[-1], y0 = full_fig.layout.yaxis.range[0],
                  x1=list(data['Close'].index)[-1], y1 = full_fig.layout.yaxis.range[1],
                  line=dict(color="black", width=1, dash="dash"),row=1,col=1)

    fig.update_layout(height=600, width=1400) #, title_text="Bitcoin Price Prediction")

    # delete y axis in Trend & Seasonality
    fig.update_yaxes(showticklabels=False, row=1,col=2)
    fig.update_yaxes(showticklabels=False, row=2,col=2)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True) # 테두리
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True) # 테두리
        
    return fig