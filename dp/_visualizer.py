"""This file provides the visualizer for the DPArray class."""
import json
from enum import IntEnum

import dash
import numpy as np
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html
from plotly.colors import get_colorscale, sample_colorscale

from dp._logger import Op


class CellType(IntEnum):
    """
    CellType determines the color of elements in the DP array.

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

    Args:
        indices(set): Set of indices. It is expected that the indices are
        integers for 1D arrays and tuples of to integers for 2D arrays.

    Returns:
        formatted_indices: outputs the given indices in numpy form:
        a list of values on the first dimension and a list of values on
        the second dimension.
    """
    # ignore if 1-d or no indicies
    if len(indices) <= 0 or isinstance(list(indices)[0], int):
        return list(indices)

    x, y = [], []
    for i in indices:
        x.append(i[0])
        y.append(i[1])
    return x, y


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


def create_figure(dp_arr, start=0, show=True, colorscale_name="Sunset", row_labels=None, column_labels=None):
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
        contents[_index_set_to_numpy_index(
            arr_data[Op.WRITE])] = CellType.WRITE
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
    values = np.where(np.isnan(values.astype(float)), "", values)
    heatmaps = [
        go.Heatmap(
            z=color,
            x=column_labels,
            y=row_labels,
            text=val,
            texttemplate="%{text}",
            textfont={"size": 20},
            customdata=hovertext[i],
            hovertemplate="<b>%{y} %{x}</b><br>%{customdata}" +
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

    # Create the figure
    fig = go.Figure(
        data=heatmaps[start],
        layout=go.Layout(
            title=dp_arr.array_name,
            title_x=0.5,
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
    fig.update_layout(clickmode="event+select")

    return fig, values, heatmaps


def display(dp_arr,
            start=0,
            show=True,
            colorscale_name="Sunset",
            row_labels=None,
            column_labels=None):
    """Creates an interactive display of the given DPArray in a webpage.

    Using a slider and buttons for time travel. This UI will have interactive
    testing as well as the figure.

    Args:
        dp_arr (DPArray): DParray to be visualized.
        start (int): Starting interation to be displayed. Defaults to 0.
        show (bool): Whether to show figure. Defaults to true.
        colorscale_name (str): Name of built-in colorscales in plotly. See
            plotly.colors.named_colorscales for the built-in colorscales.
        row_labels (list of str): Row labels of the DP array.
        column_labels (list of str): Column labels of the DP array.

    Returns:
        Plotly figure: Figure of DPArray as it is filled out by the recurrence.
    """

    if not isinstance(dp_arr, list):
        dp_arr = [dp_arr]

    graphs = []
    heatmaps = []
    fig, values, heatmap = create_figure(dp_arr[0], colorscale_name=colorscale_name,
                        row_labels=row_labels, column_labels=column_labels)
    graphs.append(dcc.Graph(id="graph", figure=fig))
    heatmaps.append(heatmap)
    for i, arr in enumerate(dp_arr[1:]):
        fig, _, heatmap = create_figure(arr, colorscale_name=colorscale_name,
                            row_labels=row_labels, column_labels=column_labels)
        graphs.append(dcc.Graph(id="graph_2", figure=fig))
        heatmaps.append(heatmap)

    styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

    # Create Dash App
    app = Dash()

    # Creates layout for dash app
    app.layout = html.Div([
        *graphs,
        dcc.Slider(min=0,
                   max=len(values) - 1,
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
        current_heatmap = heatmaps[0][value]

        # Update the figure data
        existing_figure["data"] = [current_heatmap]

        return existing_figure

    # Callback to change current heatmap based on slider value
    @app.callback(Output("graph_2", "figure"), [Input("my_slider", "value")],
                  [State("graph_2", "figure")])
    def update_figure_2(value, existing_figure):
        # Get the heatmap for the current slider value
        current_heatmap = heatmaps[1][value]

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
            current_value = min(current_value + 1, len(values) - 1)
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
            return -1  # Runs interval indefinitely
        if "stop" in ctx.triggered_id:
            return 0  # Stops interval from running

    # Changes value of slider based on state of play/stop button
    @app.callback(Output("my_slider", "value", allow_duplicate=True),
                  Input("interval", "n_intervals"),
                  State("my_slider", "value"),
                  prevent_initial_call=True)
    def button_iterate_slider(n_intervals, value):
        new_value = (value + 1) % (len(values))
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
            z_value = click_data["points"][0]["text"]
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

    @app.callback(Output('click-data', 'children'), Input('graph', 'clickData'))
    def display_click_data(clickData):
        return json.dumps(clickData, indent=2)

    if show:
        app.run_server(debug=True, use_reloader=True)

    return figure


# TODO:
# def backtrack(dp_arr, indices, direction="forward"):
#     pass
# Backtracking:
# backtrack(OPT, indices_in_order, function)
# indices_in_order = [(5, 10), (4, 9), ...]
# indices_in_order = [(4, 9), (5, 10), ...]
# function = lambda x,y: return x-y
# function((5, 10), (4,9)) months
# template = "{} months"
