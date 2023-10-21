"""This file provides the visualizer for the DPArray class."""
import numpy as np
import plotly.graph_objs as go
import dash
from dash import Dash, html, dcc, Output, Input, State
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

    # Getting the data values for each frame
    arr = np.array([t[dp_arr.array_name]["contents"] for t in timesteps])

    # Plotly heatmaps requires 2d input as data.
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 1)

    # Creates a hovertext array with the same shape as arr.
    # For each frame and cell in arr, populate the corresponding hovertext
    # cell with its value and dependencies.
    # TODO: Highlight will probably be handled here.
    hovertext = np.full_like(arr, None)
    for t, record in enumerate(timesteps):
        for write_idx in record[dp_arr.array_name][Op.WRITE]:
            # Fill in corresponding hovertext cell with value and dependencies
            # Have to add a dimension if arr is a 1D Array
            if isinstance(write_idx, int):
                hovertext[t:, 0, write_idx] = (
                    f"Value: {arr[t, 0, write_idx]}<br />Dependencies: "
                    f"{record[dp_arr.array_name][Op.READ] or '{}'}")
            else:
                hovertext[(np.s_[t:], *write_idx)] = (
                    f"Value: {arr[(t, *write_idx)]}<br />Dependencies: "
                    f"{record[dp_arr.array_name][Op.READ] or '{}'}")

    # Create heatmaps.
    # NOTE: We should be using "customdata" for hovertext.
    heatmaps = [
        go.Heatmap(
            z=z,
            text=hovertext[i],
            texttemplate="%{z}",
            textfont={"size": 20},
            hovertemplate="<b>%{x} %{y}</b><br>%{text}" + "<extra></extra>",
            zmin=0,
            zmax=100,
            colorscale=theme,
            xgap=1,
            ygap=1,
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

    # Create the figure
    fig = go.Figure(
        data=heatmaps[start],
        layout=go.Layout(
            title=fig_title,
            title_x=0.5,
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
    fig.update_layout(clickmode="event+select")

    styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

    # Create Dash App
    app = Dash()

    # Creates layout for dash app
    app.layout = html.Div([
        dcc.Graph(id="graph", figure=fig),
        dcc.Slider(min=0,
                   max=len(arr) - 1,
                   step=1,
                   value=0,
                   updatemode="drag",
                   id="my_slider"),
        dcc.Store(id="store-keypress", data=0),
        dcc.Interval(id="interval",
                     interval=1000,
                     n_intervals=0,
                     max_intervals=0),
        html.Button("Dash_Play", id="play"),
        html.Button("Dash_Stop", id="stop"),
        html.Div([
            dcc.Markdown("""
                **SELF-TESTING**
            """),
            html.Pre(id="click-data", style=styles["pre"]),
        ],
                 className="three columns"),
        dcc.Input(id="user_input", type="number", placeholder="",
                  debounce=True),
        html.Div(id="user_output"),
        dcc.Store(id="store-clicked-z"),
        html.Div(id="comparison-result")
    ])

    # Callback to change current heatmap based on slider value
    @app.callback(Output("graph", "figure"), [Input("my_slider", "value")],
                  [State("graph", "figure")])
    def update_figure(value, existing_figure):
        # Get the heatmap for the current slider value
        current_heatmap = heatmaps[value]

        # Update the figure data
        existing_figure["data"] = [current_heatmap]

        return existing_figure

    # Update slider value baed on store-keypress.
    # Store-keypress is changed in assets/custom.js
    @app.callback(Output("my_slider", "value"), Input("store-keypress", "data"),
                  State("my_slider", "value"))
    def update_slider(key_data, current_value):
        if key_data == 37:  # left arrow
            current_value = max(current_value - 1, 0)
        elif key_data == 39:  # right arrow
            current_value = min(current_value + 1, len(arr) - 1)
        return current_value

    # Starts and stop interval from running 
    @app.callback(Output("interval", "max_intervals"),
                  [Input("play", "n_clicks"),
                   Input("stop", "n_clicks")], State("interval",
                                                     "max_intervals"))
    def control_interval(start_clicks, stop_clicks, max_intervals):
        ctx = dash.callback_context
        if not ctx.triggered_id:
            return dash.no_update
        if "play" in ctx.triggered_id:
            return -1  #Runs interval indefinitely
        elif "stop" in ctx.triggered_id:
            return 0  #Stops interval from running

    # Changes value of slider based on state of play/stop button
    @app.callback(Output("my_slider", "value", allow_duplicate=True),
                  Input("interval", "n_intervals"),
                  State("my_slider", "value"),
                  prevent_initial_call=True)
    def button_iterate_slider(n_intervals, value):
        new_value = (value + 1) % (len(arr))
        return new_value

    # Displays user input after pressing enter
    @app.callback(
        Output("user_output", "children"),
        Input("user_input", "value"),
    )
    def update_output(user_input):
        return f"User Input: {user_input}"

    # Saves data of clicked element inside of store-clicked-z
    @app.callback(
        [Output("store-clicked-z", "data"),
         Output("user_input", "value")], Input("graph", "clickData"))
    def save_click_data(click_data):
        if click_data is not None:
            z_value = click_data["points"][0]["z"]
            return {"z_value": z_value}, ""
        return dash.no_update, dash.no_update

    # Tests if user input is correct
    # TODO: Change what it compares the user input to
    @app.callback(
        Output("comparison-result", "children"),
        [Input("user_input", "value"),
         Input("store-clicked-z", "data")])
    def compare_input_and_click(user_input, click_data):
        if user_input is None or click_data is None:
            return dash.no_update
        z_value = click_data.get("z_value", None)
        if z_value is None:
            return "No point clicked yet."

        # Converting to integers before comparison
        try:
            if int(user_input) == int(z_value):
                return "Correct!"
            return f"Incorrect. The clicked z-value is {z_value}."
        except ValueError:
            return ""

    if show:
        app.run_server(debug=True, use_reloader=True)

    return fig
