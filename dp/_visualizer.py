import numpy as np
import streamlit as st
from dp._logger import Logger, Op
import copy
import pdb
import plotly.graph_objs as go

def display(dp_arr, num_timesteps, recurrence=None, title=None):
    """Creates an interactive display the given DPArray in a streamlit webpage.
    Using a slider and buttons for time travel.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        max_timesteps (int): Maximum number of time steps to be visualized.
        recurrence (str): Markdown string of the recurrence relation of the DP. Optional.
        title (str): String for the title of the webpage. Optional.
    """
    #defining containers
    header = st.container()
    recurrence = st.container()
    plot_spot = st.empty()
    buttons = st.empty()
    # Setting up streamlit session state with variable frame
    if "now" not in st.session_state:
        st.session_state.now = 0

    # If there is a title create a container for it.
    if (title):
        header.title(title)
        st.sidebar.markdown(title)

    if (recurrence):    
        recurrence.markdown("Recurrence: " + recurrence)


def _display_dp(dp_arr, n, start=0, theme='solar'):
    """Plots the dp array as a plotly graph object animation.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        max_timesteps (int): Maximum number of time steps to be visualized.
        recurrence (str): Markdown string of the recurrence relation of the DP. Optional.
        title (str): String for the title of the webpage. Optional.
    """
    # Obtaining the dp_array timesteps object
    dp_arr_timesteps = dp_arr.get_timesteps()

    # Getting the data values for each frame
    arr = [timestep[dp_arr._array_name]['contents'] for timestep in dp_arr_timesteps]
    arr = np.array(arr)
    arr = arr.reshape(len(arr), 1, n)

    # Writing in the hovertext for each frame -- this and the data values should be their own functions
    # Source: https://stackoverflow.com/questions/45569092/plotly-python-heatmap-change-hovertext-x-y-z
    # Hover for each cell should look like "val: {cell_value}" and "references: {cells referenced to obtain current value}"
    hovertext = np.full(arr.shape, None)
    for timestep, changed_arrays in enumerate(dp_arr_timesteps):
        for written_cell in changed_arrays[dp_arr._array_name][Op.WRITE]:
            if timestep > 0:
                hovertext[timestep][0][:written_cell] = hovertext[timestep-1][0][:written_cell]
            hovertext[timestep][0][written_cell] = 'Value: {}<br />Dependencies: {}'.format(arr[timestep][0][written_cell], changed_arrays[dp_arr._array_name][Op.READ])

    # Rendering all the frames for the animation
    frames = [go.Frame(data=[go.Heatmap(z=arr[i], colorscale=theme, hoverinfo='text', text=hovertext[i], zmin=0, zmax=100)], name=f'Frame {i}') for i in range(len(arr))]

    # Create steps for the slider
    steps = []
    for i in range(len(arr)):
        step = dict(
            args=[[f'Frame {i}'], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
            label=str(i),
            method="animate",
        )
        steps.append(step)

    # Create the slider
    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(font=dict(size=20), prefix="Frame:", visible=True, xanchor="right"),
        transition=dict(duration=300, easing="cubic-in-out"),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]

    # Create the figure
    fig = go.Figure(
        data=[go.Heatmap(z=arr[0], colorscale=theme, hoverinfo='text', text=hovertext[i], zmin=0, zmax=100)],
        layout=go.Layout(
            title="Frame 0",
            title_x=0.5,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                        dict(label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                    ])],
            sliders=sliders
        ),
        frames=frames
    )

    fig.show()