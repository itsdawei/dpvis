import numpy as np
import plotly.graph_objs as go

from dp._logger import Op


def display(dp_arr, starting_timestep=0, theme="viridis", show=True):
    """Creates an interactive display the given DPArray in a streamlit webpage.
    Using a slider and buttons for time travel. This UI will have interactive
    testing as well as the figure.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        starting_timestep (int): Starting iteration to be displayed. Defaults
            to 0.
        theme (str): Colorscheme of heatmap. Defaults to simple_white.
        show (str): Boolean to control whether to show figure. Defaults to true.

    Returns:
        Plotly figure: Figure of DPArray as it is filled out by the recurrence.
    """
    figure = _display_dp(dp_arr,
                         start=starting_timestep,
                         theme=theme,
                         show=show)
    return figure


def _parse_timestep_list(timestep_list, dp_array_name):
    """Parses the timesteps list to create two lists containing the cells and the hovertext.

    Args:
        timestep_list (list): The timestep list of a DPArray.
        dp_array_name (str): Name of array being tracked for graphing.
        TODO: dp_array_name should be a list of strs in the future.

    Returns:
        tuple of value data and hovertext data for each frame
    """

    # Getting the data values for each frame
    arr = np.array([t[dp_array_name]["contents"] for t in timestep_list])

    # Plotly heatmaps requires 2d input as data.
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 1)

    # Creates a hovertext array with the same shape as arr.
    # For each frame and cell in arr, populate the corresponding hovertext
    # cell with its value and dependencies.
    # TODO: Highlight will probably be handled here.
    hovertext = np.full_like(arr, None)
    for t, record in enumerate(timestep_list):
        for write_idx in record[dp_array_name][Op.WRITE]:
            # Fill in corresponding hovertext cell with value and dependencies
            # Have to add a dimension if arr is a 1D Array
            if isinstance(write_idx, int):
                hovertext[t:, 0, write_idx] = (
                    f"Value: {arr[t, 0, write_idx]}<br />Dependencies: "
                    f"{record[dp_array_name][Op.READ] or '{}'}")
            else:
                hovertext[(np.s_[t:], *write_idx)] = (
                    f"Value: {arr[(t, *write_idx)]}<br />Dependencies: "
                    f"{record[dp_array_name][Op.READ] or '{}'}")

    return arr, hovertext


def _display_dp(dp_arr,
                fig_title="DP Array",
                start=0,
                theme="solar",
                show=True):
    """Plots the dp array as an animated heatmap.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        fig_title (str): Title of the figure.
        start (int): Starting interation to be displayed. Defaults to 0.
        theme (str): Theme of heatmap. Defaults to solar.
        show (bool): Whether to show figure. Defaults to true.
    """
    # Obtaining the dp_array timesteps object.
    timesteps = dp_arr.get_timesteps()

    # Parse the timesteps list to get the data values and hovertext data for each frame.
    arr, hovertext = _parse_timestep_list(timesteps, dp_arr.array_name)

    # Create heatmaps.
    # NOTE: We should be using "customdata" for hovertext.
    heatmaps = [
        go.Heatmap(
            z=z,
            text=hovertext[i],
            texttemplate="%{z}",
            textfont={"size": 20},
            hovertemplate="<b>%{x} %{y}</b><br>" + "%{text}" +
            "<extra></extra>",
            zmin=0,
            zmax=100,
            colorscale=theme,
        ) for i, z in enumerate(arr)
    ]

    # Rendering all the frames for the animation.
    frames = [
        go.Frame(name=f"Frame {i}", data=heatmap)
        for i, heatmap in enumerate(heatmaps)
    ]

    # Create steps for the slider
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
    } for i in range(len(arr))]

    # Create the slider
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

    # Create the figure
    fig = go.Figure(
        data=heatmaps[start],
        layout=go.Layout(
            title=fig_title,
            title_x=0.5,
            updatemenus=[go.layout.Updatemenu(type="buttons", buttons=buttons)],
            sliders=sliders,
            xaxis={
                "tickmode": "array",
                "tickvals": np.arange(arr.shape[2]),
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                "tickmode": "array",
                "tickvals": np.arange(arr.shape[1]),
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
