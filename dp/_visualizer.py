"""This file provides the visualizer for the DPArray class."""
import json
from enum import IntEnum

import dash
import numpy as np
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.colors import get_colorscale, sample_colorscale

from dp import DPArray
from dp._index_converter import _indices_to_np_indices
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

    Using a slider and buttons for time travel. This UI has interactive
    testing as well as the figure.

    Args:
        array (DPArray): DParray to be visualized.
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

        self._primary_name = None
        self._graph_metadata = {}
        """
        {
        "array_name": {
                arr:,
                t_figure:, 
                t_dependency:, 
                t_highlight:, 
                t_heatmap:,
            }
        }
        """

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
            raise TypeError("Array must be DPArray")

        # First array is the primary array.
        if self._primary_name is None:
            self._primary_name = arr.array_name

        if len(self._arrays) > 0:
            logger = self._arrays[0].logger
            for a in self._arrays:
                if logger is not a.logger:
                    raise ValueError("Added arrays should have the same"
                                     "logger")

        self._graph_metadata[arr.array_name] = {
            "arr": arr,
            "figure_kwargs": {
                "column_labels": column_labels,
                "row_labels": row_labels,
                "colorscale_name": colorscale_name,
            }
        }

    def _parse_timesteps(self, arr):
        timesteps = arr.get_timesteps()
        t = len(timesteps)

        # Height and width of the array.
        if len(arr.shape) == 1:
            h, w = *arr.shape, 1
        else:
            h, w = arr.shape

        # Obtaining the dp_array timesteps object.
        timesteps = arr.get_timesteps()

        name = arr.array_name

        # Constructing the color and value matrix for each timestep.
        t_color_matrix = np.empty((t, h, w))
        t_value_matrix = np.empty((t, h, w))
        t_read_matrix = np.empty((t, h, w), dtype="object")
        t_write_matrix = np.empty((t, h, w), dtype="object")
        t_highlight_matrix = np.empty((t, h, w), dtype="object")
        modded = []
        for i, timestep in enumerate(timesteps):
            t_arr = timestep[name]
            c_mat = np.copy(t_arr["contents"])
            mask = np.isnan(c_mat.astype(float))
            c_mat[np.where(mask)] = CellType.EMPTY
            c_mat[np.where(~mask)] = CellType.FILLED
            c_mat[_indices_to_np_indices(t_arr[Op.READ])] = CellType.READ
            c_mat[_indices_to_np_indices(t_arr[Op.WRITE])] = CellType.WRITE
            c_mat[_indices_to_np_indices(
                t_arr[Op.HIGHLIGHT])] = CellType.HIGHLIGHT
            t_color_matrix[i] = c_mat
            t_value_matrix[i] = t_arr["contents"]
            modded.append(list(t_arr[Op.WRITE]))

            for write_idx in t_arr[Op.WRITE]:
                if isinstance(write_idx, int):
                    write_idx = (0, write_idx)
                indices = (np.s_[i:], *write_idx)
                # Fill in corresponding hovertext cell with value and dependencies.
                # An added dimension is needed if t_arr is a 1D Array.
                t_read_matrix[indices] = t_arr[Op.READ]
                t_highlight_matrix[indices] = t_arr[Op.HIGHLIGHT]

        t_color_matrix = np.array(t_color_matrix)
        t_value_matrix = np.array(t_value_matrix)

        # Plotly heatmaps requires 2d input as data.
        if t_value_matrix.ndim == 2:
            t_color_matrix = np.expand_dims(t_color_matrix, 1)
            t_value_matrix = np.expand_dims(t_value_matrix, 1)
            modded = [[(idx, 0) for idx in t] for t in modded]

        metadata = {
            "t_color_matrix": t_color_matrix,
            "t_value_matrix": t_value_matrix,
            "t_read_matrix": t_read_matrix,
            "t_write_matrix": t_write_matrix,
            "t_highlight_matirx": t_highlight_matrix,
            "t_modded_matrix": modded,
        }
        self._graph_metadata[arr.array_name] = {
            **self._graph_metadata[arr.array_name],
            **metadata
        }

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
        self._parse_timesteps(arr)

        t_value_matrix = self._graph_metadata[arr.array_name]["t_value_matrix"]
        t_color_matrix = self._graph_metadata[arr.array_name]["t_color_matrix"]
        t_read_matrix = self._graph_metadata[arr.array_name]["t_read_matrix"]

        if len(arr.shape) == 1:
            h, = arr.shape
            w = 1
        else:
            h, w = arr.shape

        # Extra hovertext info:
        # <br>Value: {value_text}<br>Dependencies: {deps_text}
        value_text = np.where(np.isnan(t_value_matrix.astype(float)), "",
                              t_value_matrix.astype("str"))
        deps_text = np.where(t_read_matrix == set(), "{}",
                             t_read_matrix.astype("str"))
        extra_hovertext = np.char.add("<br>Value: ", value_text)
        extra_hovertext = np.char.add(extra_hovertext, "<br>Dependencies: ")
        extra_hovertext = np.char.add(extra_hovertext, deps_text)

        # Remove extra info for empty cells.
        extra_hovertext[t_color_matrix == CellType.EMPTY] = ""

        # Create heatmaps.
        t_heatmaps = [
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
        frames = [go.Frame(data=heatmap) for heatmap in t_heatmaps]

        # Create the figure.
        column_alias = row_alias = None
        if column_labels:
            column_alias = {i: column_labels[i] for i in range(w)}
        if row_labels:
            row_alias = {i: row_labels[i] for i in range(h)}
        figure = go.Figure(
            data=t_heatmaps[0],
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

        metadata = {
            "figure": figure,
            "t_heatmaps": t_heatmaps,
        }
        self._graph_metadata[arr.array_name] = {
            **self._graph_metadata[arr.array_name],
            **metadata,
        }

    def _attach_callbacks(self):
        heatmaps = self._graph_metadata[self._primary_name]["t_heatmaps"]
        values = self._graph_metadata[self._primary_name]["t_value_matrix"]
        modded = self._graph_metadata[self._primary_name]["t_modded_matrix"]
        colors = self._graph_metadata[self._primary_name]["t_color_matrix"]

        # Callback to change current heatmap based on slider value
        @self.app.callback(Output("graph", "figure"),
                           Output("current-write", "data"),
                           Input("slider", "value"), State("graph", "figure"))
        def update_figure(value, existing_figure):
            # Get the heatmap for the current slider value.
            current_heatmap = heatmaps[value]

            # Update the figure data.
            existing_figure["data"] = [current_heatmap]

            return existing_figure, 0

        # Update slider value baed on store-keypress.
        # Store-keypress is changed in assets/custom.js.
        @self.app.callback(Output("slider", "value"),
                           Input("store-keypress", "data"),
                           State("slider", "value"))
        def update_slider(key_data, current_value):
            if key_data == 37:  # Left arrow
                current_value = max(current_value - 1, 0)
            elif key_data == 39:  # Right arrow
                current_value = min(current_value + 1, len(values) - 1)
            return current_value

        # Starts and stop interval from running.
        @self.app.callback(Output("interval", "max_intervals"),
                           Input("play", "n_clicks"), Input("stop", "n_clicks"))
        def control_interval(_start_clicks, _stop_clicks):
            ctx = dash.callback_context
            if not ctx.triggered_id:
                return dash.no_update
            if "play" in ctx.triggered_id:
                return -1  # Runs interval indefinitely.
            if "stop" in ctx.triggered_id:
                return 0  # Stops interval from running.
            return dash.no_update

        # Changes value of slider based on state of play/stop button.
        @self.app.callback(Output("slider", "value", allow_duplicate=True),
                           Input("interval", "n_intervals"),
                           State("slider", "value"),
                           State("self-testing-mode", "data"),
                           prevent_initial_call=True)
        def button_iterate_slider(_n_intervals, value, self_testing_mode):
            if not self_testing_mode:
                new_value = (value + 1) % (len(values))
                return new_value
            return value

        # Displays user input after pressing enter.
        @self.app.callback(
            Output("user-output", "children"),
            Input("user-input", "value"),
        )
        def update_output(user_input):
            return f"User Input: {user_input}"

        @self.app.callback(Output("click-data", "children"),
                           Input("graph", "clickData"))
        def display_click_data(click_data):
            return json.dumps(click_data, indent=2)

        # Define callback to toggle self_testing_mode
        @self.app.callback(Output("self-testing-mode", "data"),
                           Input("self-test-button", "n_clicks"),
                           State("self-testing-mode", "data"))
        def toggle_self_testing_mode(n_clicks, self_testing_mode):
            if n_clicks is None:
                return dash.no_update  # Do not update if the button wasn"t clicked
            return not self_testing_mode  # Toggle the state

        # Define another callback that uses self_testing_mode
        @self.app.callback(
            Output("toggle-text", "children"),
            Output(component_id="slider-container", component_property="style"),
            Input("self-testing-mode", "data"))
        def toggle_playback_and_slider(self_testing_mode):
            if self_testing_mode:
                return "Self-Testing Mode: ON", {"display": "none"}
            return "Self-Testing Mode: OFF", {"display": "block"}

        # Saves data of clicked element inside of store-clicked-z.
        @self.app.callback(Output("store-clicked-z", "data"),
                           Output("user-input", "value"),
                           Input("graph", "clickData"))
        def save_click_data(click_data):
            if click_data is not None:
                z_value = click_data["points"][0]["text"]
                return {"z_value": z_value}, ""
            return dash.no_update, dash.no_update

        # Tests if user input is correct.
        @self.app.callback(Output("comparison-result", "children"),
                           Output("current-write", "data",
                                  allow_duplicate=True),
                           Output("graph", "figure", allow_duplicate=True),
                           Input("user-input", "value"),
                           Input("self-testing-mode", "data"),
                           State("slider", "value"),
                           State("current-write", "data"),
                           State("graph", "figure"),
                           prevent_initial_call=True)
        def compare_input_and_frame(user_input, is_self_testing, current_frame,
                                    current_write, existing_figure):
            # TODO: Was the isdigit comparison necessary?
            if is_self_testing and user_input is not None and user_input != "":
                next_frame = (current_frame + 1) % len(values)
                x, y = modded[next_frame][current_write]
                test = values[next_frame][x][y]
                next_write = (current_write + 1) % len(modded[next_frame])

                if int(user_input) == int(test):
                    existing_figure["data"][0]["z"][x][y] = CellType.EMPTY
                    return "Correct!", (next_write), existing_figure
                return "Incorrect!", (current_write), existing_figure
            return dash.no_update

        @self.app.callback(Output("graph", "figure", allow_duplicate=True),
                           Input("current-write", "data"),
                           Input("slider", "value"),
                           Input("self-testing-mode", "data"),
                           State("graph", "figure"),
                           prevent_initial_call=True)
        def highlight_testing_cell(current_write, current_frame,
                                   is_self_testing, existing_figure):
            next_frame = (current_frame + 1) % len(values)
            x, y = modded[next_frame][current_write]
            if is_self_testing:
                # TODO: If we want to isolate the cell being tested, we need to remove this line
                # But, if this line is removed, then we have issues with the dependencies function.
                existing_figure["data"][0]["z"] = colors[current_frame]
                existing_figure["data"][0]["z"][x][y] = CellType.HIGHLIGHT
                return existing_figure
            # TODO: Is the following line necessary?
            # existing_figure["data"][0]["z"][x][y] = CellType.EMPTY
            return dash.no_update

        @self.app.callback(Output("graph", "figure", allow_duplicate=True),
                           Input("graph", "clickData"),
                           Input("self-testing-mode", "data"),
                           State("slider", "value"),
                           State("graph", "figure"),
                           prevent_initial_call=True)
        def display_dependencies(click_data, value, figure, self_testing_mode):
            # If in self_testing_mode or selected cell is empty, do nothing.
            if self_testing_mode or figure["data"][0]["z"][click_data["points"][0][
                    "y"]][click_data["points"][0]["x"]] == CellType.EMPTY:
                return dash.no_update
            click = click_data["points"][0]
            x = click["x"]
            y = click["y"]

            # If selected cell is empty, do nothing.
            if figure["data"][0]["z"][y][x] == CellType.EMPTY:
                return dash.no_update

            # Clear all highlight, read, and write cells to filled.
            figure["data"][0]["z"] = list(
                map(
                    lambda x: list(
                        map(
                            lambda y: CellType.FILLED
                            if y != CellType.EMPTY else y, x)),
                    figure["data"][0]["z"]))

            # Highlight selected cell.
            figure["data"][0]["z"][y][x] = CellType.WRITE

            # Highlight dependencies.
            deps = self._graph_metadata[
                self._primary_name]["t_dependency_matrix"]
            d = deps[value][y][x]
            for dy, dx in d:
                figure["data"][0]["z"][dy][dx] = CellType.READ

            # Highlight highlights.
            high = self._graph_metadata[
                self._primary_name]["t_highlight_matrix"]
            h = high[value][y][x]
            for hy, hx in h:
                figure["data"][0]["z"][hy][hx] = CellType.HIGHLIGHT

            return figure

    def show(self):
        graphs = []
        for name in self._graph_metadata:
            arr = self._graph_metadata[name]["arr"]
            kwargs = self._graph_metadata[name]["figure_kwargs"]
            self._create_figure(arr, **kwargs)

            graphs.append(
                dcc.Graph(id="graph",
                          figure=self._graph_metadata[name]["figure"]))

        styles = {
            "pre": {
                "border": "thin lightgrey solid",
                "overflowX": "scroll"
            }
        }

        max_timestep = len(
            list(self._graph_metadata.values())[0]["t_heatmaps"]) - 1

        self.app.layout = html.Div([
            graphs[0],
            html.Div(id="slider-container",
                     children=[
                         dcc.Slider(min=0,
                                    max=max_timestep,
                                    step=1,
                                    value=0,
                                    updatemode="drag",
                                    id="slider"),
                         html.Button("Play", id="play"),
                         html.Button("Stop", id="stop"),
                     ],
                     style={"display": "block"}),
            dcc.Store(id="store-keypress", data=0),
            dcc.Interval(id="interval",
                         interval=1000,
                         n_intervals=0,
                         max_intervals=0),
            html.Div([
                dcc.Markdown("""
                    **SELF-TESTING**
                """),
                html.Pre(id="click-data", style=styles["pre"]),
            ],
                     className="three columns"),
            dcc.Input(id="user-input",
                      type="number",
                      placeholder="",
                      debounce=True),
            html.Div(id="user-output"),
            dcc.Store(id="store-clicked-z"),
            html.Div(id="comparison-result"),
            html.Button("Test Myself!", id="self-test-button"),
            html.Div(id="next-prompt"),
            dcc.Store(id="self-testing-mode", data=False),
            html.Div(id="toggle-text", children="Self-Testing Mode: OFF"),
            dcc.Store(id="current-write", data=0)
        ])

        self._attach_callbacks()

        # self.app.run_server(debug=True, use_reloader=True)
        self.app.run_server(debug=False, use_reloader=True)
