import pandas as pd
import plotly.express as px


def plotly_figure_1(data, asig):
    """Creates a nice figure."""

    data.insert(loc=6, column="Reparto", value=0)
    data.insert(loc=6, column="Día", value="")
    dia = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado"]
    for i in range(len(asig)):
        for j in range(len(asig[i])):
            if asig[i][j] == 1:
                data.loc[j:j, ("Reparto")] = i
                data.loc[j:j, ("Día")] = data["Día"][j] + " " + dia[i]

    fig = px.scatter_mapbox(
        data,
        lat="lat",
        lon="lon",
        hover_name="Id_Cliente",
        hover_data=["Vol_Entrega", "Día"],
        color="Reparto",
        zoom=10,
        height=300,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig
