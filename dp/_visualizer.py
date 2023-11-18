"""This file provides the visualizer for the DPArray class."""
import copy
# import json
from enum import IntEnum

import dash
import dash_bootstrap_components as dbc
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
            description=None,
            colorscale_name="Sunset"):
    """Creates an interactive display of the given DPArray in a webpage.

    Using a slider and buttons for time travel. This UI has interactive
    testing as well as the figure.

    Args:
        array (DPArray): DParray to be visualized.
        row_labels (list of str): Row labels of the DP array.
        column_labels (list of str): Column labels of the DP array.
        description (str): Markdown of the description for the DPArray.
        colorscale_name (str): Name of built-in colorscales in plotly. See
            plotly.colors.named_colorscales for the built-in colorscales.
    """
    visualizer = Visualizer()
    visualizer.add_array(array,
                         column_labels=column_labels,
                         row_labels=row_labels,
                         description=description,
                         colorscale_name=colorscale_name)
    visualizer.show()


class Visualizer:
    """Visualizer class.

    Attributes:
        _arrays (list of DPArray): Contains the values of the DP array.
        _primary_name (string): Name of the primary array.
        _graph_metadata (dict): A dictionary of metadata for each array.
            The dictionary has the following format:
            {
                array_name: {
                        arr: ,
                        t_dependency: ,
                        t_highlight: ,
                        t_heatmap: ,
                        ...
                },
                ...
            }
    """

    def __init__(self):
        """Initialize Visualizer object."""
        self._primary = None
        self._graph_metadata = {}

        # https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
        # If we use a dark theme, make the layout background transparent
        themes = [dbc.themes.LUX]

        # Create Dash App.
        self._app = Dash(
            __name__,
            # name="dynvis: Dynamic Program Visualization",
            external_stylesheets=themes,
            prevent_initial_callbacks=True)

    def add_array(self,
                  arr,
                  column_labels=None,
                  row_labels=None,
                  description="",
                  colorscale_name="Sunset"):
        """Add a DPArray to the visualization."""
        # TODO: @David Docstrings
        if not isinstance(arr, DPArray):
            raise TypeError("Array must be DPArray")

        # First array is the primary array.
        if self._primary is None:
            self._primary = arr.array_name

        self._graph_metadata[arr.array_name] = {
            "arr": arr,
            "description": description,
            "figure_kwargs": {
                "column_labels": column_labels or [],
                "row_labels": row_labels or [],
                "colorscale_name": colorscale_name,
            }
        }

        logger = self._graph_metadata[self._primary]["arr"].logger
        if logger is not arr.logger:
            raise ValueError("Added arrays should have the same"
                             "logger")

    def _parse_timesteps(self, arr):
        """Parse the timesteps of the logger."""
        timesteps = arr.get_timesteps()
        t = len(timesteps)

        name = arr.array_name

        # Height and width of the array.
        if len(arr.shape) == 1:
            h, w = 1, *arr.shape
            # Convert 1D timestep to 2D timestep.
            for i, timestep in enumerate(timesteps):
                t_arr = timestep[name]
                t_arr["contents"] = np.expand_dims(t_arr["contents"], 0)
                for op in Op:
                    new_op_coords = {(0, idx) for idx in t_arr[op]}
                    t_arr[op] = new_op_coords
        else:
            h, w = arr.shape

        # Constructing the color and value matrix for each timestep.
        # Initializes to CellType.EMPTY
        t_color_matrix = np.zeros((t, h, w))
        t_value_matrix = np.empty((t, h, w))
        # For each cell, stores its dependency.
        t_read_matrix = np.empty((t, h, w), dtype="object")
        t_highlight_matrix = np.empty((t, h, w), dtype="object")
        # Boolean mask of which cell is written to at timestep t.
        t_write_matrix = np.zeros((t, h, w), dtype="bool")
        for i, timestep in enumerate(timesteps):
            t_arr = timestep[name]
            mask = np.isnan(t_arr["contents"].astype(float))
            t_color_matrix[i][np.nonzero(~mask)] = CellType.FILLED
            t_color_matrix[i][_indices_to_np_indices(
                t_arr[Op.READ])] = CellType.READ
            t_color_matrix[i][_indices_to_np_indices(
                t_arr[Op.WRITE])] = CellType.WRITE
            t_color_matrix[i][_indices_to_np_indices(
                t_arr[Op.HIGHLIGHT])] = CellType.HIGHLIGHT
            t_value_matrix[i] = t_arr["contents"]

            for write_idx in t_arr[Op.WRITE]:
                indices = (np.s_[i:], *write_idx)
                t_read_matrix[indices] = t_arr[Op.READ]
                t_highlight_matrix[indices] = t_arr[Op.HIGHLIGHT]
                t_write_matrix[i][write_idx] = True

        return {
            "t_color_matrix": t_color_matrix,
            "t_value_matrix": t_value_matrix,
            "t_read_matrix": t_read_matrix,
            "t_write_matrix": t_write_matrix,
            "t_highlight_matrix": t_highlight_matrix,
        }

    def _show_figure_trace(self, figure, i):
        """Make exactly one trace of the figure visible.

        Args:
            figure (plotly.go.Figure): The ith trace from this figure will be
                visible, while all the other traces will be hidden.
            i (int): The index of the trace that will be shown.

        Returns:
            plotly.go.Figure: Figure after the trace is shown.
        """
        return figure.update_traces(visible=False).update_traces(visible=True,
                                                                 selector=i)

    def _create_figure(self, arr, colorscale_name="Sunset"):
        """Create a figure for an array.

        Args:
            arr (DPArray): DParray to be visualized.
            show (bool): Whether to show figure. Defaults to true.
            colorscale_name (str): Name of built-in colorscales in plotly. See
                plotly.colors.named_colorscales for the built-in colorscales.

        Returns:
            plotly.go.figure: Figure of DPArray as it is filled out by the
                recurrence.
        """
        name = arr.array_name
        self._graph_metadata[name].update(self._parse_timesteps(arr))

        metadata = self._graph_metadata[name]
        kwargs = metadata["figure_kwargs"]

        t_value_matrix = metadata["t_value_matrix"]
        t_color_matrix = metadata["t_color_matrix"]
        t_read_matrix = metadata["t_read_matrix"]

        h, w = t_value_matrix.shape[1], t_value_matrix.shape[2]

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

        # Create the figure.
        # column_alias = row_alias = None
        column_alias = dict(enumerate(kwargs["column_labels"]))
        row_alias = dict(enumerate(kwargs["row_labels"]))
        figure = go.Figure(
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
                "coloraxis": {
                    "showscale": False
                },
                "clickmode": "event+select",
            })

        for color, val, extra in zip(t_color_matrix, value_text,
                                     extra_hovertext):
            figure.add_heatmap(
                z=color,
                text=val,
                texttemplate="%{text}",
                textfont={"size": 20},
                customdata=extra,
                hovertemplate="<b>%{y}, %{x}</b>%{customdata}<extra></extra>",
                **_get_colorbar_kwargs(colorscale_name),
                xgap=1,
                ygap=1,
                visible=False,
            )

        return self._show_figure_trace(figure, 0)

    def _attach_callbacks(self):
        """Attach callbacks."""
        values = self._graph_metadata[self._primary]["t_value_matrix"]
        t_write_matrix = self._graph_metadata[self._primary]["t_write_matrix"]
        main_figure = self._graph_metadata[self._primary]["figure"]

        output_figure = [
            Output(name, "figure", allow_duplicate=True)
            for name in self._graph_metadata
        ]

        @self.app.callback(output_figure, Input("slider", "value"))
        def update_figure(t):
            """Update each graph based on the slider value."""
            return [
                self._show_figure_trace(metadata["figure"], t)
                for metadata in self._graph_metadata.values()
            ]

        @self.app.callback(
            Output("slider", "value"),
            Input("store-keypress", "data"),
            Input("interval", "n_intervals"),
            State("slider", "value"),
        )
        def update_slider(key_data, _, t):
            """Update the value of slider based on state of play/stop button.

            Update slider value based on store-keypress. Store-keypress is
            changed in assets/custom.js.
            """
            if ctx.triggered_id == "interval":
                return (t + 1) % len(values)
            if key_data in [37, 39]:
                return (t + key_data - 38) % len(values)
            return dash.no_update

        @self.app.callback(Output("interval", "max_intervals"),
                           Input("play", "n_clicks"), Input("stop", "n_clicks"),
                           Input("self-test-button", "n_clicks"))
        def play_pause_playback(_start_clicks, _stop_clicks, _n_clicks):
            """Starts and stop playback from running.

            Pauses the playback when "stop" or "self-test-button" is pressed.
            """
            if ctx.triggered_id == "play":
                return -1  # Runs interval indefinitely.
            if ctx.triggered_id in ["stop", "self-test-button"]:
                return 0  # Stops interval from running.
            return dash.no_update

        # @self.app.callback(Output("click-data", "children"),
        #                    Input(self._primary, "clickData"))
        # def display_click_data(click_data):
        #     # TODO: Remove this
        #     return json.dumps(click_data, indent=2)

        @self.app.callback(
            Output("test-mode-toast", "is_open"),
            Output(component_id="playback-control", component_property="style"),
            Input("test-info", "data"))
        def toggle_layout(info):
            if info["test_mode"]:
                return True, {"visibility": "hidden"}
            return False, {"visibility": "visible"}

        @self.app.callback(
            Output("test-info", "data"),
            Input("self-test-button", "n_clicks"),
            State("test-info", "data"),
            State("slider", "value"),
        )
        def toggle_test_mode(_, info, t):
            """Toggles self-testing mode.

            Args:
                n_clicks (int): This callback is triggered by clicking the
                    self-test-button component.
                t (int): The current timestep retrieved from the slider
                    component.
            """
            # No tests to be performed on the last timestep.
            if t == len(values):
                # TODO: notify user that there is no more testing
                return dash.no_update

            if info["test_mode"]:
                return {
                    "test_mode": False,
                    "cur_test": 0,
                    "num_tests": -1,
                }
            return {
                "test_mode": True,
                "cur_test": 0,
                "num_tests": np.count_nonzero(t_write_matrix[t + 1]),
            }

        @self.app.callback(
            Output(self._primary, "figure", allow_duplicate=True),
            Input("test-info", "data"), State("slider", "value"))
        def highlight_tests(info, t):
            if not info["test_mode"]:
                return self._show_figure_trace(main_figure, t)

            fig = copy.deepcopy(main_figure)
            z = fig.data[t].z

            # Highlight the cell that is being tested on.
            cur_test = info["cur_test"]
            x, y = np.transpose(np.nonzero(t_write_matrix[t + 1]))[cur_test]
            z[x][y] = CellType.WRITE

            return fig.update_traces(z=z, selector=t)

        @self.app.callback(
            Output("correct", "is_open"),
            Output("incorrect", "is_open"),
            Output("test-info", "data", allow_duplicate=True),
            # Trigger this callback every time "enter" is pressed.
            Input("user-input", "n_submit"),
            State("user-input", "value"),
            State("test-info", "data"),
            State("slider", "value"),
        )
        def compare_input_and_frame(_, user_input, info, t):
            """Tests if user input is correct."""
            if not info["test_mode"]:
                return dash.no_update

            cur_test = info["cur_test"]
            x, y = np.transpose(np.nonzero(t_write_matrix[t + 1]))[cur_test]
            test = values[t + 1][x][y]

            if user_input == test:
                info["cur_test"] += 1
                info["test_mode"] = info["cur_test"] < info["num_tests"]
                return True, False, info
            return False, True, dash.no_update

        @self.app.callback(
            Output(self._primary, "figure", allow_duplicate=True),
            Input(self._primary, "clickData"), State("test-info", "data"),
            State("slider", "value"))
        def display_dependencies(click_data, info, t):
            # Skip this callback in testing mode.
            if info["test_mode"]:
                return dash.no_update

            x = click_data["points"][0]["x"]
            y = click_data["points"][0]["y"]

            fig = copy.deepcopy(main_figure)
            z = fig.data[t].z

            # If selected cell is empty, do nothing.
            if z[y][x] == CellType.EMPTY:
                return dash.no_update

            # Clear all highlight, read, and write cells to filled.
            z[z != CellType.EMPTY] = CellType.FILLED

            # Highlight selected cell.
            z[y][x] = CellType.WRITE

            # Highlight dependencies.
            d = self._graph_metadata[self._primary]["t_read_matrix"]
            z[_indices_to_np_indices(d[t][y][x])] = CellType.READ

            # Highlight highlights.
            h = self._graph_metadata[self._primary]["t_highlight_matrix"]
            z[_indices_to_np_indices(h[t][y][x])] = CellType.HIGHLIGHT

            return fig.update_traces(z=z, selector=t)

    def show(self):
        """Visualizes the DPArrays.

        Create the figures for each DPArray, attach the callbacks, and render
        the graph.
        """
        graphs = []
        for name, metadata in self._graph_metadata.copy().items():
            arr = metadata["arr"]
            figure = self._create_figure(arr)
            graphs.append(dcc.Graph(id=name, figure=figure))
            self._graph_metadata[name]["figure"] = figure

        max_timestep = len(self._graph_metadata[self._primary]["figure"].data)

        questions = [
            "What is the next cell?",
            "What are its dependencies?",
            "What is its value?",
        ]

        test_select_checkbox = dbc.Row([
            dbc.Col(
                dbc.Checklist(
                    questions,
                    questions,
                    id="test-select-checkbox",
                )),
            dbc.Col(dbc.Button("Test Myself!",
                               id="self-test-button",
                               class_name="h-100",
                               color="info"),
                    width="auto")
        ])

        description_md = [
            dcc.Markdown(metadata["description"],
                         mathjax=True,
                         className="border border-primary")
            for metadata in self._graph_metadata.values()
        ]

        alerts = [
            dbc.Alert("You are in self-testing mode",
                      id="test-mode-toast",
                      is_open=False,
                      color="info",
                      style={
                          "position": "fixed",
                          "bottom": 10,
                          "left": 10,
                          "width": 350,
                      }),
            dbc.Alert("Correct!",
                      id="correct",
                      is_open=False,
                      color="success",
                      duration=3000,
                      fade=True,
                      className="alert-auto position-fixed w-25",
                      style={
                          "bottom": 10,
                          "left": 10,
                          "z-index": 9999,
                      }),
            dbc.Alert("Incorrect!",
                      id="incorrect",
                      is_open=False,
                      color="danger",
                      duration=3000,
                      fade=True,
                      className="alert-auto position-fixed w-25",
                      style={
                          "bottom": 10,
                          "left": 10,
                          "z-index": 9999,
                      })
        ]

        sidebar = html.Div([
            dbc.Stack([
                *description_md,
                test_select_checkbox,
                dbc.Input(id="user-input", type="number", placeholder=""),
            ],
                      id="sidebar",
                      className="border border-warning"),
        ])

        playback_control = [
            dbc.Col(dbc.Button("Play", id="play"), width="auto"),
            dbc.Col(dbc.Button("Stop", id="stop"), width="auto"),
            dbc.Col(dcc.Slider(
                min=0,
                max=max_timestep - 1,
                step=1,
                value=0,
                updatemode="drag",
                id="slider",
            )),
            dcc.Interval(id="interval",
                         interval=1000,
                         n_intervals=0,
                         max_intervals=0),
        ]

        datastores = [
            dcc.Store(id="store-keypress", data=0),
            dcc.Store(id="test-info",
                      data={
                          "test_mode": False,
                          "num_tests": -1,
                          "cur_test": 0
                      }),
        ]

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(sidebar, width="auto"),
                        dbc.Col([
                            dbc.Row(
                                playback_control,
                                id="playback-control",
                                class_name="g-0",
                                align="center",
                            ),
                            dbc.Row(
                                dbc.Stack(graphs),
                                id="page-content",
                                className="border border-warning",
                            )
                        ])
                    ],
                    class_name="g-3"),
                *alerts,
                *datastores,
            ],
            fluid=True,
        )

        self._attach_callbacks()

        self.app.run_server(debug=True, use_reloader=True)
        # self.app.run_server(debug=False, use_reloader=True)

    @property
    def app(self):
        """Returns the Dash app object."""
        return self._app
