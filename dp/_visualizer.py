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
                t_arr["cell_annotations"] = {
                    (0, idx): annotation
                    for idx, annotation in t_arr["cell_annotations"].items()
                }
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
        # List of dictionaries. Empty if no annotations.
        # t_annotations: [{array_name: annotation}, ...]
        # t_cell_annotations: [{array_name: {cell_idx: annotation}}, ...]
        t_annotations = np.fill(len(timesteps), {})
        t_cell_annotations = np.fill(len(timesteps), {})
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
            t_annotations[i] = t_arr["annotations"]
            t_cell_annotations[i] = t_arr["cell_annotations"]

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
            "t_annotations": t_annotations,
            "t_cell_annotations": t_cell_annotations,
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
        # (if cell annotation present:) <br>Note: {annotation}
        value_text = np.where(np.isnan(t_value_matrix.astype(float)), "",
                              t_value_matrix.astype("str"))
        deps_text = np.where(t_read_matrix == set(), "{}",
                             t_read_matrix.astype("str"))
        extra_hovertext = np.char.add("<br>Value: ", value_text)
        extra_hovertext = np.char.add(extra_hovertext, "<br>Dependencies: ")
        extra_hovertext = np.char.add(extra_hovertext, deps_text)

        # Remove extra info for empty cells.
        extra_hovertext[t_color_matrix == CellType.EMPTY] = ""

        # Add any annotations to the hovertext.
        cell_annotation_text = metadata["t_cell_annotations"]
        for t, timestep in enumerate(cell_annotation_text):
            for j, cell_annotation in timestep.items():
                extra_hovertext[t][j] += "<br>" + cell_annotation

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
                "hoverlabel": {
                    "namelength": -1,
                },
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
        t_annotations = [[
            html.P(
                f"{name}: {arr['t_annotations'][t]}",
                style={
                    "textAlign": "center",
                    "width": "auto"
                },
            )
            for name, arr in self._graph_metadata.items()
            if arr["t_annotations"][t]
        ]
                         for t in range(len(values))]

        t_read_matrix = self._graph_metadata[self._primary]["t_read_matrix"]

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

        # update annotation toast based on slider value
        @self.app.callback(Output("array-annotation", "style"),
                           Output("array-annotation", "children"),
                           Input("slider", "value"))
        def update_annotation(t):
            """Update the annotation toast based on the slider value."""
            if not t_annotations[t]:
                return {"display": "none"}, []

            return {"display": "block"}, t_annotations[t]

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
            Output(component_id="playback-control", component_property="style"),
            Input("test-info", "data"))
        def toggle_layout(info):
            if info["tests"]:
                return {"visibility": "hidden"}
            return {"visibility": "visible"}

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

            # Turn off testing mode.
            if info["tests"]:
                return {
                    "tests": [],
                }

            # Create list of write indices for t+1.
            write_mask = t_write_matrix[t + 1]
            all_writes = np.transpose(np.nonzero(write_mask))

            # Create list of dependencies for t+1.
            # Any all writes have the same reads on the same timestep, so we
            # arbitrarily pick the first one.
            all_reads = list(t_read_matrix[t + 1][write_mask][0])

            # TODO: Populate according to radio box.
            test_q = []

            # Filling out write test.
            # The truth list is a list of indices that are written to in the
            # next cell.
            test_q.append({
                "truth": all_writes,
                "render": [],
                "color": CellType.WRITE,
                "expected_triggered_id": self._primary,
                "tip": "What cells are written to in the next frame? (Click "
                       "in any order)"
            })

            # Filling out read test.
            test_q.append({
                "truth": all_reads,
                "render": [],
                "color": CellType.READ,
                "expected_triggered_id": self._primary,
                "tip": "What cells are read for the next timestep? (Click "
                       "in any order)"
            })

            # Filling out all value tests.
            for x, y in zip(*np.nonzero(write_mask)):
                test_q.append({
                    "truth": [values[t + 1][x][y]],
                    "render": [(x, y)],
                    "color": CellType.WRITE,
                    "expected_triggered_id": "user-input",
                    "tip": f"What is the value of cell ({x}, {y})?"
                })

            return {"tests": test_q}

        @self.app.callback(
            Output(self._primary, "figure", allow_duplicate=True),
            Output("test-instructions", "children"),
            Input("test-info", "data"),
            State("slider", "value"),
        )
        def display_tests(info, t):
            alert = dbc.Alert(is_open=False,
                              color="danger",
                              class_name="alert-auto position-fixed w-25")
            if not info["tests"]:
                return self._show_figure_trace(main_figure, t), alert

            fig = copy.deepcopy(main_figure)

            # Clear HIGHLIGHT, READ, and WRITE cells to FILLED.
            z = fig.data[t].z.astype("bool").astype("int")

            # Highlight the revelant cells as specified by "render".
            test = info["tests"][0]
            render = test["render"]
            for x, y in render:
                z[x][y] = test["color"]

            # Bring up test-specific instructions.
            alert.is_open = True
            alert.children = test["tip"]

            return fig.update_traces(z=z, selector=t), alert

        @self.app.callback(
            Output("correct", "is_open"),
            Output("incorrect", "is_open"),
            Output("test-info", "data", allow_duplicate=True),
            # For manually resetting clickData.
            Output(self._primary, "clickData"),
            # Trigger this callback every time "enter" is pressed.
            Input("user-input", "n_submit"),
            Input(self._primary, "clickData"),
            State("user-input", "value"),
            State("test-info", "data"),
        )
        def validator(_, click_data, user_input, info):
            """Tests if user input is correct."""
            if not info["tests"]:
                return dash.no_update

            if ctx.triggered_id != info["tests"][0]["expected_triggered_id"]:
                # Wrong type of input: no update.
                return dash.no_update

            if ctx.triggered_id == self._primary:
                # Click on graph.
                answer = [
                    click_data["points"][0]["y"],
                    click_data["points"][0]["x"],
                ]
            else:
                # Enter number.
                answer = user_input

            test = info["tests"][0]
            truths = test["truth"]

            is_correct = False
            # Check that [x, y] is a row of truths.
            if answer in truths:
                # Remove from truth and update render the test values.
                truths.remove(answer)
                info["tests"][0]["render"].append(answer)
                is_correct = True

            # If all truths have been found, pop test from queue.
            if not truths:
                info["tests"].pop(0)

            return is_correct, not is_correct, info, None

        @self.app.callback(
            Output(self._primary, "figure", allow_duplicate=True),
            Input(self._primary, "clickData"), State("test-info", "data"),
            State("slider", "value"))
        def display_dependencies(click_data, info, t):
            # Skip this callback in testing mode.
            if info["tests"] or not click_data:
                return dash.no_update

            x = click_data["points"][0]["x"]
            y = click_data["points"][0]["y"]

            fig = copy.deepcopy(main_figure)
            z = fig.data[t].z

            # If selected cell is empty, do nothing.
            if z[y][x] == CellType.EMPTY:
                return dash.no_update

            # Clear HIGHLIGHT, READ, and WRITE cells to FILLED.
            z = z.astype("bool").astype("int")

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
                      }),
            html.Div(id="test-instructions",
                     style={
                         "bottom": 80,
                         "left": 10,
                         "z-index": 9999,
                     }),
        ]

        array_annotations = dbc.Card(
            dbc.CardBody(
                [],
                id="array-annotation",
            ),
            color="info",
            outline=True,
        )

        sidebar = html.Div([
            dbc.Stack([
                *description_md,
                test_select_checkbox,
                dbc.Input(id="user-input", type="number", placeholder=""),
                array_annotations,
            ],
                      id="sidebar",
                      className="border border-warning"),
        ])

        playback_control = [
            dbc.Col(dbc.Button("Play", id="play"), width="auto"),
            dbc.Col(dbc.Button("Stop", id="stop"), width="auto"),
            dbc.Col(
                dcc.Slider(
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
            dcc.Store(
                id="test-info",
                data={
                    # [W, V1, V2, ..., Vn, R]
                    # Each element is the test states for the current timestep.
                    # - W: Click on all writes.
                    # - Vi: Entered the value for the ith write.
                    # - R: Click on all reads.
                    "tests": [],
                }),
        ]

        self.app.layout = dbc.Container(
            [
                dbc.Row([
                    dbc.Col(sidebar, width=4),
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
                        ),
                    ],
                            width=8),
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
