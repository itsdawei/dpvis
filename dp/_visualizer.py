import numpy as np
import streamlit as st
from dp._logger import Logger, Op
import copy
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

    # st.slider("Iteration #:",
    #           0,
    #           num_timesteps - 1,
    #           key="now",
    #           on_change=_change_options(arr, col_name))

# Very suboptimal -- A private function that converts a DPArray object to an array suitable for inputting to the heatmap.
# In the future, we should remove this function and just make the heatmap work with the original DPArray object.
def _DPArray_to_Array(dp_obj, n):
    A = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        A[i:][i] = dp_obj[i]
    return A


def _display_dp(dp_arr, n, start=0):
    """Plots the dp array as a plotly graph object animation.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        max_timesteps (int): Maximum number of time steps to be visualized.
        recurrence (str): Markdown string of the recurrence relation of the DP. Optional.
        title (str): String for the title of the webpage. Optional.
    """
    arr = _DPArray_to_Array(dp_arr, n)
    print(arr)
    fig = go.Figure(
        data=[go.Heatmap(z=arr[start])],
        layout=go.Layout(
            title="Frame 0",
            title_x=0.5,
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None]),
                        dict(label="Pause",
                             method="animate",
                             args=[None,
                                   {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                            )])]
        ),
        frames=[go.Frame(data=[go.Heatmap(z=arr[i])],
                        layout=go.Layout(title_text=f"Frame {i}")) 
                for i in range(1, n)]
    )
    fig.show()

    # # Base code
    # N = 50
    # M = np.random.random((N, 10, 10))
    # fig = go.Figure(
    #     data=[go.Heatmap(z=M[0])],
    #     layout=go.Layout(
    #         title="Frame 0",
    #         title_x=0.5,
    #         updatemenus=[dict(
    #             type="buttons",
    #             buttons=[dict(label="Play",
    #                         method="animate",
    #                         args=[None]),
    #                     dict(label="Pause",
    #                         method="animate",
    #                         args=[None,
    #                             {"frame": {"duration": 0, "redraw": False},
    #                                 "mode": "immediate",
    #                                 "transition": {"duration": 0}}],
    #                         )])]
    #     ),
    #     frames=[go.Frame(data=[go.Heatmap(z=M[i])],
    #                     layout=go.Layout(title_text=f"Frame {i}")) 
    #             for i in range(1, N)]
    # )

    # fig.show()