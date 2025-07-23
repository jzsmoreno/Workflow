import plotly.express as px
import plotly.graph_objects as go


def plot_series(df_series, data_pred, i=0):
    df_plot = df_series.copy()
    df_plot["prediction"] = data_pred[i, : len(df_plot)]

    # Define los colores
    color_map = {str(i): "blue", "prediction": "red"}

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["prediction"],
            mode="lines",
            name="prediction",
            line=dict(width=2, dash="dashdot", color=color_map["prediction"]),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot[i],
            mode="lines",
            name=str(i),
            line=dict(width=3, dash="solid", color=color_map[str(i)]),
        )
    )

    fig.update_layout(
        title=f"Serie real '{i}' vs Prediction", xaxis_title="Index", yaxis_title="Valor"
    )

    return fig


def plot_time_series(df, i=0):
    fig = px.line(df.iloc[:, i])
    return fig


def plot_displot(df_series, data_pred, i=0):
    df_series["prediction"] = data_pred[i, : len(df_series)]
    # plotly figure
    values = [i, -1]
    fig = px.histogram(df_series, y=df_series.iloc[:, values].columns, marginal="box")
    return fig
