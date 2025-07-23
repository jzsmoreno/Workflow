import plotly.express as px


def plot_series(df_series, data_pred, i=0):
    df_plot = df_series.copy()
    df_plot["prediction"] = data_pred[i, : len(df_plot)]

    cols_a_melt = [i, "prediction"]

    df_long = df_plot.reset_index().melt(
        id_vars="index", var_name="Serie", value_name="Valor", value_vars=cols_a_melt
    )

    color_map = {str(i): "blue", "prediction": "red"}  # Real series color  # Prediction color

    fig = px.line(
        df_long,
        x="index",
        y="Valor",
        color="Serie",
        title=f"Serie real '{i}' vs Prediction",
        color_discrete_map=color_map,
    )
    fig.for_each_trace(
        lambda trace: (
            trace.update(line=dict(width=3))
            if trace.name == str(i)
            else trace.update(line=dict(width=2, dash="dot"))
        )
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
