"""This file provides the visualizer for the DPArray class."""
import collections.abc
import json
from enum import IntEnum

import dash
import numpy as np
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.colors import get_colorscale, sample_colorscale

from dp import DPArray
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
    # Ignore if 1-d or no indicies.
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

    # Colorscale for the colorbar.
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


def display(array,
            row_labels=None,
            column_labels=None,
            colorscale_name="Sunset"):
    """Creates an interactive display of the given DPArray in a webpage.

    Using a slider and buttons for time travel. This UI will have interactive
    testing as well as the figure.

    Args:
        array (DPArray): The array to be visualized.
        row_labels (list of str): Row labels of the DP array.
        column_labels (list of str): Column labels of the DP array.
        colorscale_name (str): Name of built-in colorscales in plotly. See
            plotly.colors.named_colorscales for the built-in colorscales.
    """
    visualizer = Visualizer()
    visualizer.add_array(array,
                         column_labels=column_labels,
                         row_labels=row_labels,
                         colorscale_name=colorscale_name)
    visualizer.show()


class Visualizer:

    def __init__(self):
        self._arrays = []
        self._figure_kwargs = []

        # Create Dash App.
        self._app = Dash()

    @property
    def app(self):
        """Returns the Dash app object."""
        return self._app

    def add_array(self,
                  arr,
                  column_labels=None,
                  row_labels=None,
                  colorscale_name="Sunset"):
        if not isinstance(arr, DPArray):
            raise TypeError()
        # TODO: Check that arr have same logger.
        self._arrays.append(arr)
        self._figure_kwargs.append({
            "column_labels": column_labels,
            "row_labels": row_labels,
            "colorscale_name": colorscale_name,
        })

    def _parse_timesteps(self, arr):
        timesteps = arr.get_timesteps()

        w, = arr.shape
        h = 1
        name = arr.array_name
        t = len(timesteps)

        # Constructing the color and value matrix for each timestep.
        t_color_matrix = np.empty((t, h, w))
        t_value_matrix = np.empty((t, h, w))
        t_read_matrix = np.empty((t, h, w), dtype="object")
        t_write_matrix = np.empty((t, h, w), dtype="object")
        t_highlight_matrix = np.empty((t, h, w), dtype="object")
        for i, timestep in enumerate(timesteps):
            arr = timestep[name]
            c_mat = np.copy(arr["contents"])
            mask = np.isnan(c_mat.astype(float))
            c_mat[np.where(mask)] = CellType.EMPTY
            c_mat[np.where(~mask)] = CellType.FILLED
            c_mat[_index_set_to_numpy_index(arr[Op.READ])] = CellType.READ
            c_mat[_index_set_to_numpy_index(arr[Op.WRITE])] = CellType.WRITE
            c_mat[_index_set_to_numpy_index(
                arr[Op.HIGHLIGHT])] = CellType.HIGHLIGHT
            t_color_matrix[i] = c_mat
            t_value_matrix[i] = arr["contents"]

            for write_idx in arr[Op.WRITE]:
                if isinstance(write_idx, int):
                    write_idx = (0, write_idx)
                indices = (np.s_[i:], *write_idx)
                # Fill in corresponding hovertext cell with value and dependencies.
                # An added dimension is needed if arr is a 1D Array.
                t_read_matrix[indices] = timestep[name][Op.READ]
                t_highlight_matrix[indices] = timestep[name][Op.HIGHLIGHT]

        t_color_matrix = np.array(t_color_matrix)
        t_value_matrix = np.array(t_value_matrix)

        # Plotly heatmaps requires 2d input as data.
        if t_value_matrix.ndim == 2:
            t_color_matrix = np.expand_dims(t_color_matrix, 1)
            t_value_matrix = np.expand_dims(t_value_matrix, 1)

        # TODO: Clean this up
        return t_color_matrix, t_value_matrix, t_read_matrix, t_write_matrix, t_highlight_matrix

    def _create_figure(self,
                       arr,
                       colorscale_name="Sunset",
                       row_labels=None,
                       column_labels=None):
        """Create a figure for an array.

        Args:
            arr (DPArray): DParray to be visualized.
            show (bool): Whether to show figure. Defaults to true.
            colorscale_name (str): Name of built-in colorscales in plotly. See
                plotly.colors.named_colorscales for the built-in colorscales.
            row_labels (list of str): Row labels of the DP array.
            column_labels (list of str): Column labels of the DP array.

        Returns:
            Plotly figure: Figure of DPArray as it is filled out by the recurrence.
        """
        (
            t_color_matrix,
            t_value_matrix,
            dependency_matrix,
            _,
            highlight_matrix,
        ) = self._parse_timesteps(arr)

        w, = arr.shape
        h = 1

        # Extra hovertext info:
        # <br>Value: {value_text}<br>Dependencies: {deps_text}
        value_text = np.where(np.isnan(t_value_matrix.astype(float)), "",
                              t_value_matrix.astype("str"))
        deps_text = np.where(dependency_matrix == set(), "{}",
                             dependency_matrix.astype("str"))
        extra_hovertext = np.char.add("<br>Value: ", value_text)
        extra_hovertext = np.char.add(extra_hovertext, "<br>Dependencies: ")
        extra_hovertext = np.char.add(extra_hovertext, deps_text)

        # Remove extra info for empty cells.
        extra_hovertext[t_color_matrix == CellType.EMPTY] = ""

        # Create heatmaps.
        heatmaps = [
            go.Heatmap(
                z=color,
                text=val,
                texttemplate="%{text}",
                textfont={"size": 20},
                customdata=extra,
                hovertemplate="<b>%{y}, %{x}</b>%{customdata}<extra></extra>",
                **_get_colorbar_kwargs(colorscale_name),
                xgap=1,
                ygap=1,
            ) for color, val, extra in zip(t_color_matrix, value_text,
                                           extra_hovertext)
        ]

        # Rendering all the frames for the animation.
        frames = [go.Frame(data=heatmap) for heatmap in heatmaps]

        # Create the figure.
        column_alias = row_alias = None
        if column_labels is not None:
            column_alias = {i: column_labels[i] for i in range(w)}
        if row_labels is not None:
            row_alias = {i: row_labels[i] for i in range(h)}
        figure = go.Figure(
            data=heatmaps[0],
            frames=frames,
            layout={
                "title": arr.array_name,
                "title_x": 0.5,
                "xaxis": {
                    "tickmode": "array",
                    "tickvals": np.arange(w),
                    "labelalias": column_alias,
                    "showgrid": False,
                    "zeroline": False,
                },
                "yaxis": {
                    "tickmode": "array",
                    "tickvals": np.arange(h),
                    "labelalias": row_alias,
                    "showgrid": False,
                    "zeroline": False
                },
            },
        )
        figure.update_coloraxes(showscale=False)
        figure.update_layout(clickmode="event+select")

        return {
            "figure": figure,
            "heatmaps": heatmaps,
            "dependency_matrix": dependency_matrix,
            "highlight_matrix": highlight_matrix,
        }

    # TODO: Maybe the callbacks should be in a differnt file --- dash_callbacks.py?
    def _attach_callbacks(self, max_timestep):
        graph_callbacks = {
            "output": [],
            "state": [],
            "input": [],
        }
        graph_frames = []
        for arr in self._arrays:
            graph_callbacks["output"].append(Output(arr.array_name, "figure"))
            graph_callbacks["input"].append(Input(arr.array_name, "clickData"))
            graph_callbacks["state"].append(State(arr.array_name, "figure"))
            graph_frames.append(
                self.app.layout[arr.array_name].figure["frames"])

        # Callback to update all heatmaps to slider timestep.
        @self.app.callback(
            graph_callbacks["output"],
            Input("slider", "value"),
            graph_callbacks["state"],
        )
        def update_figure(t, *figures):
            for i, frame in enumerate(graph_frames):
                # Get the heatmap for the current slider value.
                current_heatmap = frame[t]["data"][0]

                # Update the figure data.
                figures[i]["data"] = [current_heatmap]
            return figures

        # BUG: This doesn't work for some reason.
        # Update slider value based on store-keypress.
        # Store-keypress is changed in assets/custom.js.
        @self.app.callback(Output("slider", "value"),
                           Input("store-keypress", "data"),
                           State("slider", "value"))
        def update_slider(key_data, current_value):
            if key_data == 37:  # Left arrow
                current_value = max(current_value - 1, 0)
            elif key_data == 39:  # Right arrow
                current_value = min(current_value + 1, max_timestep)
            return current_value

        # Starts and stop interval from running.
        @self.app.callback(
            Output("interval", "max_intervals"),
            [Input("play", "n_clicks"),
             Input("stop", "n_clicks")], State("interval", "max_intervals"))
        def control_interval(_start_clicks, _stop_clicks, _max_intervals):
            triggered_id = ctx.triggered_id
            if triggered_id == "play":
                return -1  # Runs interval indefinitely.
            if triggered_id == "stop":
                return 0  # Stops interval from running.

        # Changes value of slider based on state of play/stop button.
        @self.app.callback(Output("slider", "value", allow_duplicate=True),
                           Input("interval", "n_intervals"),
                           State("slider", "value"),
                           prevent_initial_call=True)
        def button_iterate_slider(_n_intervals, value):
            return (value + 1) % max_timestep

        # Displays user input after pressing enter.
        @self.app.callback(
            Output("user_output", "children"),
            Input("user_input", "value"),
        )
        def update_output(user_input):
            return f"User Input: {user_input}"

        # Tests if user input is correct.
        # TODO: Change what it compares the user input to
        @self.app.callback(
            Output("comparison-result", "children"),
            [Input("user_input", "value"),
             Input("store-clicked-z", "data")])
        def compare_input_and_click(user_input, click_data):
            if user_input is None or click_data is None:
                return dash.no_update
            z_value = click_data.get("z_value", None)
            if z_value is None:
                return "No point clicked yet."

            # Converting to integers before comparison.
            try:
                if int(user_input) == int(z_value):
                    return "Correct!"
                return f"Incorrect. The clicked z-value is {z_value}."
            except ValueError:
                return ""

        @self.app.callback(Output("click-data", "children"),
                           graph_callbacks["input"],
                           prevent_initial_call=True)
        def display_click_data(*click_datum):
            return json.dumps(click_datum, indent=2)

        # TODO: Allow multiple graphs.
        @self.app.callback(
            Output('dp_array', 'figure',
                   allow_duplicate=True), [Input('dp_array', 'clickData')],
            [State('slider', 'value'),
             State("dp_array", "figure")],
            prevent_initial_call=True)
        def display_dependencies(click_data, value, figure):
            # If selected cell is empty, do nothing.
            if figure["data"][0]['z'][click_data["points"][0]['y']][
                    click_data["points"][0]['x']] == CellType.EMPTY:
                return dash.no_update

            # Clear all highlight, read, and write cells to filled.
            figure['data'][0]['z'] = list(
                map(
                    lambda x: list(
                        map(
                            lambda y: CellType.FILLED
                            if y != CellType.EMPTY else y, x)),
                    figure['data'][0]['z']))

            # Highlight selected cell.
            figure["data"][0]['z'][click_data["points"][0]['y']][
                click_data["points"][0]['x']] = CellType.WRITE

            # Highlight dependencies.
            dependencies = dependency_matrix[value][
                click_data["points"][0]['y']][click_data["points"][0]['x']]
            for dy, dx in dependencies:
                figure["data"][0]['z'][dy][dx] = CellType.READ

            # Highlight highlights.
            highlights = highlight_matrix[value][click_data["points"][0]['y']][
                click_data["points"][0]['x']]
            for hy, hx in highlights:
                figure["data"][0]['z'][hy][hx] = CellType.HIGHLIGHT

            return figure

    def show(self):
        graphs = []
        graph_heatmaps = []
        for arr, kwargs in zip(self._arrays, self._figure_kwargs):
            fig_dict = self._create_figure(arr, **kwargs)
            graphs.append(
                dcc.Graph(id=arr.array_name, figure=fig_dict["figure"]))
            graph_heatmaps.append(fig_dict["heatmaps"])

        styles = {
            "pre": {
                "border": "thin lightgrey solid",
                "overflowX": "scroll"
            }
        }

        self.app.layout = html.Div([
            *graphs,
            dcc.Slider(min=0,
                       max=len(graph_heatmaps[0]) - 1,
                       step=1,
                       value=0,
                       updatemode="drag",
                       id="slider"),
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
            dcc.Input(id="user_input",
                      type="number",
                      placeholder="",
                      debounce=True),
            html.Div(id="user_output"),
            dcc.Store(id="store-clicked-z"),
            html.Div(id="comparison-result")
        ])

        self._attach_callbacks(len(graph_heatmaps[0]))

        self.app.run_server(debug=True, use_reloader=True)
