"""This file provides the visualizer for the DPArray class."""
import copy
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

        # Create Dash App.
        self._app = Dash(name="dynvis: Dynamic Program Visualization",
                         prevent_initial_callbacks=True)

    def add_array(self,
                  arr,
                  column_labels=None,
                  row_labels=None,
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
            "figure_kwargs": {
                "column_labels": column_labels,
                "row_labels": row_labels,
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
            Plotly figure: Figure of DPArray as it is filled out by the
                recurrence.
        """
        name = arr.array_name
        self._graph_metadata[name].update(self._parse_timesteps(arr))

        t_value_matrix = self._graph_metadata[name]["t_value_matrix"]
        t_color_matrix = self._graph_metadata[name]["t_color_matrix"]
        t_read_matrix = self._graph_metadata[name]["t_read_matrix"]

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

        return figure

    def _attach_callbacks(self):
        """Attach callbacks."""
        values = self._graph_metadata[self._primary]["t_value_matrix"]
        t_read_matrix = self._graph_metadata[self._primary]["t_read_matrix"]
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
                metadata["figure"].frames[t]
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

        @self.app.callback(Output("click-data", "children"),
                           Input(self._primary, "clickData"))
        def display_click_data(click_data):
            # TODO: Remove this
            return json.dumps(click_data, indent=2)

        @self.app.callback(
            Output("toggle-text", "children"),
            Output(component_id="slider-container", component_property="style"),
            Input("test-info", "data"))
        def toggle_layout(info):
            if info["test_mode"]:
                return "Self-Testing Mode: ON", {"display": "none"}
            return "Self-Testing Mode: OFF", {"display": "block"}

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
            fig = copy.deepcopy(main_figure.frames[t])

            if not info["test_mode"]:
                return fig

            # Highlight the cell that is being tested on.
            cur_test = info["cur_test"]
            x, y = np.transpose(np.nonzero(t_write_matrix[t + 1]))[cur_test]
            fig.data[0]["z"][x][y] = CellType.WRITE

            return fig

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

            fig = copy.deepcopy(main_figure.frames[t])

            # If selected cell is empty, do nothing.
            if fig.data[0]["z"][y][x] == CellType.EMPTY:
                return dash.no_update

            # Clear all highlight, read, and write cells to filled.
            z = fig.data[0]["z"]
            z[z != CellType.EMPTY] = CellType.FILLED

            # Highlight selected cell.
            z[y][x] = CellType.WRITE

            # Highlight dependencies.
            d = self._graph_metadata[self._primary]["t_read_matrix"]
            z[_indices_to_np_indices(d[t][y][x])] = CellType.READ

            # Highlight highlights.
            h = self._graph_metadata[self._primary]["t_highlight_matrix"]
            z[_indices_to_np_indices(h[t][y][x])] = CellType.HIGHLIGHT

            return fig
        
        @self.app.callback(
            Output("num-tests","data"),
            Output("dep-set","data"),
            Output("tests-set","data"),
            Output("got-correct-value","data"),
            Output("got-correct-write","data"),
            Output("current-test","data"),
            Input("slider","value"),
        )
        def change_test_data_with_frame(t):
            w = self._graph_metadata[self._primary]["t_write_matrix"]
            num_tests = np.count_nonzero(t_write_matrix[t + 1])
            return num_tests, [], [], False, False, (-1,-1)
        
        @self.app.callback(
            Output("testing-prompt", "children"),
            Input("got-correct-write","data"),
            Input("got-correct-value", "data"),
            State("test-info","data"),
            State("tests-set", "data"),
            State("num-tests","data")
        )
        def change_testing_prompt(got_write, got_value, info, tests_set,num_tests):
            if not info["test_mode"]:
                return ""
            
            if not got_write:
                return "Click on the next cell to be filled in!"
            
            if not got_value:
                return "Enter the value of the next cell!"
            
            if len(tests_set) == num_tests:
                import pdb;
                pdb.set_trace()
                return "You have solved everything for the next time-step!"
            
        @self.app.callback(
            Output("got-correct-write", "data", allow_duplicate=True),
            Output("current-test","data", allow_duplicate=True),
            Input(self._primary, "clickData"),
            State("test-info","data"),
            State("got-correct-write","data"),
            State("tests-set", "data"),
            State("slider", "value"),
        )
        def test_next_write(click_data, info, got_write,tests_set,t):
            if not info["test_mode"] or got_write:
                return dash.no_update, dash.no_update

            y = click_data["points"][0]["x"]
            x = click_data["points"][0]["y"]

            writes = np.transpose(np.nonzero(t_write_matrix[t+1]))
            l = [tuple(i) for i in writes]
            tests_set_l = [tuple(i) for i in tests_set]

            if (x,y) in l and  not ((x,y) in tests_set_l):
                return True, (x,y)
            return False, dash.no_update
        
        @self.app.callback(
            Output("dep-set", "data", allow_duplicate = True),
            Output("dep-test", "children"),
            Input(self._primary, "clickData"),
            State("dep-set", "data"),
            State("test-info","data"),
            State("got-correct-write","data"),
            State("slider","value"),
            State("current-test", "data"),
            State("skip-flag","data")
        )
        def test_dep(click_data, dep_set, info, got_write,t,curr_test,skip_flag):
            if not got_write or not info["test_mode"] or skip_flag:  
                return dep_set, dash.no_update
            
            y = click_data["points"][0]["x"]
            x = click_data["points"][0]["y"]

            reads = (t_read_matrix[t+1][curr_test[0]][curr_test[1]])
            l = [tuple(i) for i in reads]
            dep_set_l = [tuple(i) for i in dep_set]

            if (x,y) in l and not ((x,y) in dep_set_l):
                dep_set.append((x,y))
                return dep_set, "Correct dependency clicked!"
            if (x,y) in l and (x,y) in dep_set_l:
                return dep_set, "Already clicked this dependency!"
            
            return dep_set, "Incorrect Dependency clicked!"
        
        @self.app.callback(
            Output("got-correct-value", "data",allow_duplicate=True),
            Input("user-input","n_submit"),
            State("user-input","value"),
            State("slider","value"),
            State("test-info","data"),
            State("current-test", "data"),
            State("skip-flag","data")
        )
        def test_value(_, user_input, t,info, curr_test,skip_flag):
            """Tests if user input is correct."""
            if not info["test_mode"] or skip_flag:
                return dash.no_update

            cur_test = info["cur_test"]
            x = curr_test[0]
            y = curr_test[1]
            test = values[t + 1][x][y]
            # import pdb;
            # pdb.set_trace()

            if user_input == test:
                return True

            return False
        


    def show(self):
        """Visualizes the DPArrays.

        Create the figures for each DPArray, attach the callbacks, and render
        the graph.
        """
        graphs = []
        for name, metadata in self._graph_metadata.copy().items():
            arr = metadata["arr"]
            figure = self._create_figure(arr, **metadata["figure_kwargs"])
            graphs.append(dcc.Graph(id=name, figure=figure))
            self._graph_metadata[name]["figure"] = figure

        max_timestep = len(self._graph_metadata[self._primary]["figure"].frames)

        self.app.layout = html.Div([
            *graphs,
            html.Div(id="slider-container",
                     children=[
                         dcc.Slider(min=0,
                                    max=max_timestep - 1,
                                    step=1,
                                    value=0,
                                    updatemode="drag",
                                    id="slider"),
                         html.Button("Play", id="play"),
                         html.Button("Stop", id="stop"),
                     ],
                     style={"display": "block"}),
            dcc.Interval(id="interval",
                         interval=1000,
                         n_intervals=0,
                         max_intervals=0),
            html.Div([
                dcc.Markdown("""
                    **SELF-TESTING**
                """),
                html.Pre(id="click-data",
                         style={
                             "border": "thin lightgrey solid",
                             "overflowX": "scroll"
                         }),
            ],
                     className="three columns"),
            dcc.Input(id="user-input",
                      type="number",
                      placeholder="",
                      debounce=True),
            html.Div(id="comparison-result"),
            html.Button("Test Myself!", id="self-test-button"),
            html.Div(id="next-prompt"),
            html.Div(id="toggle-text", children="Self-Testing Mode: OFF"),
            dcc.Store(id="store-keypress", data=0),
            dcc.Store(id="store-clicked-z"),
            dcc.Store(id="test-info",
                      data={
                          "test_mode": False,
                          "num_tests": -1,
                          "cur_test": 0
                      }),
            dcc.Store(id="current-test",data=(-1,-1)),
            dcc.Store(id="num-tests",data=0),
            dcc.Store(id="got-correct-value", data=False),
            dcc.Store(id="got-correct-write", data=False),
            dcc.Store(id="dep-set", data=[]),
            dcc.Store(id="tests-set",data=[]),
            html.Div(id="testing-prompt", children = ""),
            html.Div(id="dep-test", children=""),
            dcc.Store(id="skip-flag",data=True)
        ])

        self._attach_callbacks()

        self.app.run_server(debug=True, use_reloader=True)
        # self.app.run_server(debug=False, use_reloader=True)

    @property
    def app(self):
        """Returns the Dash app object."""
        return self._app
