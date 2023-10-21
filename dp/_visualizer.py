"""This file provides the visualizer for the DPArray class."""
from enum import IntEnum

import numpy as np
import plotly.graph_objs as go
from plotly.colors import get_colorscale, sample_colorscale

from dp._logger import Op


class CellType(IntEnum):
    """CellType determines the color of elements in the DP array.

    EMPTY and FILLED are always white and grey, respectively. The other colors
    are defined by a builtin colorscale of plotly (defaults to Sunset).
    """
    EMPTY = 0
    FILLED = 1
    HIGHLIGHT = 2
    READ = 3
    WRITE = 4


def _index_set_to_numpy_index(indices):
    """Get a set of tuples representing indices and convert it into numpy
    indicies.

    Example input: {(0, 1), (2, 3), (4, 5)}
    Example output: {[0, 2, 4], [1, 3, 5]}
    """
    # ignore if 1-d or no indicies
    if len(indices) <= 0 or isinstance(list(indices)[0], int):
        return list(indices)

    x, y = [], []
    for i in indices:
        x.append(i[0])
        y.append(i[1])
    return (x, y)


def _get_colorbar_kwargs(name):
    """Get colorscale for the DP array visualization.

    Args:
        name (str): Name of built-in colorscales in plotly. See
            named_colorscales for the built-in colorscales.

    Returns:
        dict: kwargs for the colorbar.
    """
    n = len(CellType)
    x = np.linspace(0, 1, n + 1)

    # Round the linspace to account for Python FP error.
    x = np.round(x, decimals=5)

    val = np.repeat(x, 2)[1:-1]

    # Assign colors for each cell type.
    color = sample_colorscale(get_colorscale(name), samplepoints=n)
    color[0] = "rgb(255,255,255)"  # white
    color[1] = "rgb(220,220,220)"  # grey

    # colorscale for the colorbar
    color = np.repeat(color, 2)

    return {
        "zmin": 0,
        "zmax": n,
        "colorscale": list(zip(val, color)),
        "colorbar": {
            "orientation": "h",
            "ticklabelposition": "inside",
            "tickvals": np.array(list(CellType)) + 0.5,
            "ticktext": [c.name for c in CellType],
            "tickfont": {
                "color": "black",
                "size": 20,
            },
            "thickness": 20,
        }
    }


def display(dp_arr, starting_timestep=0, show=True, colorscale_name="Sunset"):
    """Creates an interactive display the given DPArray in a streamlit webpage.

    Using a slider and buttons for time travel. This UI will have interactive
    testing as well as the figure.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        starting_timestep (int): Starting iteration to be displayed. Defaults
            to 0.
        show (str): Boolean to control whether to show figure. Defaults to true.
        colorscale_name (str): Name of built-in colorscales in plotly. See
            plotly.colors.named_colorscales for the built-in colorscales.

    Returns:
        Plotly figure: Figure of DPArray as it is filled out by the recurrence.
    """
    figure = _display_dp(dp_arr,
                         start=starting_timestep,
                         show=show,
                         colorscale_name=colorscale_name)
    return figure


def _display_dp(dp_arr,
                fig_title="DP Array",
                start=0,
                show=True,
                colorscale_name="Sunset"):
    """Plots the dp array as an animated heatmap.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        fig_title (str): Title of the figure.
        start (int): Starting interation to be displayed. Defaults to 0.
        show (bool): Whether to show figure. Defaults to true.
        colorscale_name (str): Name of built-in colorscales in plotly. See
            plotly.colors.named_colorscales for the built-in colorscales.
    """
    # Obtaining the dp_array timesteps object.
    timesteps = dp_arr.get_timesteps()

    # Getting the data values for each frame
    colors = []
    for t in timesteps:
        arr_data = t[dp_arr.array_name]
        contents = np.copy(t[dp_arr.array_name]["contents"])
        mask = np.isnan(contents.astype(float))
        contents[np.where(mask)] = CellType.EMPTY
        contents[np.where(~mask)] = CellType.FILLED
        contents[_index_set_to_numpy_index(arr_data[Op.READ])] = CellType.READ
        contents[_index_set_to_numpy_index(arr_data[Op.WRITE])] = CellType.WRITE
        contents[_index_set_to_numpy_index(
            arr_data[Op.HIGHLIGHT])] = CellType.HIGHLIGHT
        colors.append(contents)

    colors = np.array(colors)
    values = np.array([t[dp_arr.array_name]["contents"] for t in timesteps])

    # Plotly heatmaps requires 2d input as data.
    if values.ndim == 2:
        colors = np.expand_dims(colors, 1)
        values = np.expand_dims(values, 1)

    # Creates a hovertext array with the same shape as arr.
    # For each frame and cell in arr, populate the corresponding hovertext
    # cell with its value and dependencies.
    hovertext = np.full_like(values, None)
    for t, record in enumerate(timesteps):
        for write_idx in record[dp_arr.array_name][Op.WRITE]:
            # Fill in corresponding hovertext cell with value and dependencies
            # Have to add a dimension if arr is a 1D Array
            if isinstance(write_idx, int):
                hovertext[t:, 0, write_idx] = (
                    f"Value: {values[t, 0, write_idx]}<br />Dependencies: "
                    f"{record[dp_arr.array_name][Op.READ] or '{}'}")
            else:
                hovertext[(np.s_[t:], *write_idx)] = (
                    f"Value: {values[(t, *write_idx)]}<br />Dependencies: "
                    f"{record[dp_arr.array_name][Op.READ] or '{}'}")

    # Create heatmaps.
    heatmaps = [
        go.Heatmap(
            z=color,
            text=val,
            texttemplate="%{text}",
            textfont={"size": 20},
            customdata=hovertext[i],
            hovertemplate="<b>%{x} %{y}</b><br>%{customdata}" +
            "<extra></extra>",
            **_get_colorbar_kwargs(colorscale_name),
            xgap=1,
            ygap=1,
        ) for i, (val, color) in enumerate(zip(values, colors))
    ]

    # Rendering all the frames for the animation.
    frames = [
        go.Frame(name=f"Frame {i}", data=heatmap)
        for i, heatmap in enumerate(heatmaps)
    ]

    # Create steps for the slider.
    steps = [{
        "args": [[f"Frame {i}"], {
            "frame": {
                "duration": 300,
                "redraw": True
            },
            "mode": "immediate",
            "transition": {
                "duration": 300
            }
        }],
        "label": str(i),
        "method": "animate",
    } for i in range(len(values))]

    # Create the slider.
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {
                "size": 20
            },
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {
            "duration": 300,
            "easing": "cubic-in-out"
        },
        "pad": {
            "b": 10,
            "t": 50
        },
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": steps
    }]

    buttons = [{
        "label":
            "Play",
        "method":
            "animate",
        "args": [
            None, {
                "frame": {
                    "duration": 500,
                    "redraw": True
                },
                "fromcurrent": True
            }
        ]
    }, {
        "label":
            "Pause",
        "method":
            "animate",
        "args": [[None], {
            "frame": {
                "duration": 0,
                "redraw": False
            },
            "mode": "immediate",
            "transition": {
                "duration": 0
            }
        }]
    }]

    # Create the figure.
    fig = go.Figure(
        data=heatmaps[start],
        layout=go.Layout(
            title=fig_title,
            title_x=0.5,
            updatemenus=[go.layout.Updatemenu(type="buttons", buttons=buttons)],
            sliders=sliders,
            xaxis={
                "tickmode": "array",
                "tickvals": np.arange(values.shape[2]),
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                "tickmode": "array",
                "tickvals": np.arange(values.shape[1]),
                "showgrid": False,
                "zeroline": False
            },
        ),
        frames=frames,
    )
    fig.update_coloraxes(showscale=False)

    if show:
        fig.show()

    return fig
