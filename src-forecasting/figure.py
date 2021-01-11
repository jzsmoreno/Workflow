import plotly.express as px

def plot_series(df_series, data_pred, i = 0):
    df_series['prediction'] = data_pred[i, :, 0]
    # plotly figure
    values = [i, -1]
    fig = px.line(df_series, y=df_series.iloc[:, values].columns)
    return fig


def plot_time_series(df, i = 0):
    fig = px.line(df.iloc[:, i])
    return fig