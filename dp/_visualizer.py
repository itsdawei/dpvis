import numpy as np
import plotly.graph_objs as go

from dp._logger import Op


def display(dp_arr, n, starting_timestep=0, theme="solar", show=True):
    """Creates an interactive display the given DPArray in a streamlit webpage.
    Using a slider and buttons for time travel. This UI will have interactive
    testing as well as the figure.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        starting_timestep (int): Starting iteration to be displayed. Defaults
            to 0.
        theme (str): Colorscheme of heatmap. Defaults to solar.
        show (str): Boolean to control whether to show figure. Defaults to true.

    Returns:
        Plotly figure: Figure of DPArray as it is filled out by the recurrence.
    """
    figure = _display_dp(dp_arr,
                         n,
                         start=starting_timestep,
                         theme=theme,
                         show=show)
    return figure


def _display_dp(dp_arr,
                n, # pylint: disable=unused-argument
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

    # Getting the data values for each frame - should be its own function.
    arr = np.array([t[dp_arr.array_name]["contents"] for t in timesteps])

    # Plotly heatmaps requires 2d input as data.
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 1)

    # TODO: @aditya add comment to explain step-by-step what this for loop
    # does.
    # What about HIGHLIGHT?
    hovertext = np.full_like(arr, None)
    for t, record in enumerate(timesteps):
        for write_idx in record[dp_arr.array_name][Op.WRITE]:
            hovertext[t:, 0, write_idx] = (
                f"Value: {arr[t][0][write_idx]}<br />Dependencies: "
                f"{record[dp_arr.array_name][Op.READ] or '{}'}")

    # Create heatmaps.
    heatmaps = [
        go.Heatmap(
            z=z,
            colorscale=theme,
            text=hovertext[i],
            texttemplate="%{z}",
            textfont={"size": 20},
            hoverinfo="text",
            zmin=0,
            zmax=100,
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
            },
            yaxis={
                "tickmode": "array",
                "tickvals": np.arange(arr.shape[1]),
            },
        ),
        frames=frames,
    )

    if show:
        fig.show()

    return fig
