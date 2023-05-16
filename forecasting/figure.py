import plotly.express as px

def plot_series(df_series, data_pred, i = 0):
    print(len(df_series), data_pred.shape)
    df_series['prediction'] = data_pred[i, :len(df_series)]
    # plotly figure
    values = [-1, i]
    fig = px.line(df_series, y=df_series.iloc[:, values].columns)
    return fig

def plot_time_series(df, i = 0):
    fig = px.line(df.iloc[:, i])
    return fig
    
def plot_displot(df_series, data_pred, i = 0):
    df_series['prediction'] = data_pred[i, :len(df_series)]
    # plotly figure
    values = [i, -1]
    fig = px.histogram(df_series, y=df_series.iloc[:, values].columns, marginal="box")
    return fig