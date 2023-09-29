import numpy as np
import streamlit as st
from dp._logger import Op
import plotly.graph_objs as go


def display(dp_arr, n, starting_timestep=0, theme="solar", show=True):
    """Creates an interactive display the given DPArray in a streamlit webpage.
    Using a slider and buttons for time travel. This UI will have interactive 
    testing as well as the figure.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        starting_timestep (int): Starting iteration to be displayed. Defaults to 0.
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


def _display_dp(dp_arr, n, start=0, theme='solar', show=True):
    """Plots the dp array as a plotly heatmap using plotly graph object animation.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        n (int): Maximum number of time steps to be visualized.
        start (int): Starting interation to be displayed. Defaults to 0.
        theme (str): Theme of heatmap. Defaults to solar.
        show (bool): Boolean to control whether to show figure. Defaults to true.
    """
    # Obtaining the dp_array timesteps object
    dp_arr_timesteps = dp_arr.get_timesteps()

    # Getting the data values for each frame - should be its own function
    arr = [
        timestep[dp_arr._array_name]['contents']
        for timestep in dp_arr_timesteps
    ]
    arr = np.array(arr)
    arr = arr.reshape(len(arr), 1, n)

    # Writing in the hovertext for each frame -- this and the data values should be their own functions
    # Source: https://stackoverflow.com/questions/45569092/plotly-python-heatmap-change-hovertext-x-y-z
    # Hover for each cell should look like "val: {cell_value}" and "references: {cells referenced to obtain current value}"
    hovertext = np.full(arr.shape, None)
    for timestep, changed_arrays in enumerate(dp_arr_timesteps):
        for written_cell in changed_arrays[dp_arr._array_name][Op.WRITE]:
            if timestep > 0:
                hovertext[timestep][0][:written_cell] = hovertext[
                    timestep - 1][0][:written_cell]
            hovertext[timestep][0][
                written_cell] = 'Value: {}<br />Dependencies: {}'.format(
                    arr[timestep][0][written_cell],
                    changed_arrays[dp_arr._array_name][Op.READ])

    # Rendering all the frames for the animation
    frames = [
        go.Frame(data=[
            go.Heatmap(z=arr[i],
                       colorscale=theme,
                       hoverinfo='text',
                       text=hovertext[i],
                       zmin=0,
                       zmax=100)
        ],
                 name=f'Frame {i}') for i in range(len(arr))
    ]

    # Create steps for the slider
    steps = []
    for i in range(len(arr)):
        step = dict(
            args=[[f'Frame {i}'], {
                "frame": {
                    "duration": 300,
                    "redraw": True
                },
                "mode": "immediate",
                "transition": {
                    "duration": 300
                }
            }],
            label=str(i),
            method="animate",
        )
        steps.append(step)

    # Create the slider
    sliders = [
        dict(active=0,
             yanchor="top",
             xanchor="left",
             currentvalue=dict(font=dict(size=20),
                               prefix="Frame:",
                               visible=True,
                               xanchor="right"),
             transition=dict(duration=300, easing="cubic-in-out"),
             pad=dict(b=10, t=50),
             len=0.9,
             x=0.1,
             y=0,
             steps=steps)
    ]

    # Create the figure
    fig = go.Figure(data=[
        go.Heatmap(z=arr[0],
                   colorscale=theme,
                   hoverinfo='text',
                   text=hovertext[i],
                   zmin=0,
                   zmax=100)
    ],
                    layout=go.Layout(
                        title="Frame 0",
                        title_x=0.5,
                        updatemenus=[
                            dict(type="buttons",
                                 buttons=[
                                     dict(label="Play",
                                          method="animate",
                                          args=[
                                              None, {
                                                  "frame": {
                                                      "duration": 500,
                                                      "redraw": True
                                                  },
                                                  "fromcurrent": True
                                              }
                                          ]),
                                     dict(label="Pause",
                                          method="animate",
                                          args=[[None], {
                                              "frame": {
                                                  "duration": 0,
                                                  "redraw": False
                                              },
                                              "mode": "immediate",
                                              "transition": {
                                                  "duration": 0
                                              }
                                          }])
                                 ])
                        ],
                        sliders=sliders),
                    frames=frames)

    # Display figure if show is true
    if (show):
        fig.show()

    # Return figure
    return fig
