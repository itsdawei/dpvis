"""This file provides the visualizer for the DPArray class."""
import copy
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
    MAXMIN = 2
    READ = 3
    WRITE = 4


class TestType(IntEnum):
    """TestType is used to distinguish between tests in test-info.
    """
    READ = 0
    WRITE = 1
    VALUE = 2


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

    ticktext = ["EMPTY", "FILLED", "MAX/MIN", "READ", "WRITE"]
    return {
        "zmin": 0,
        "zmax": n,
        "colorscale": list(zip(val, color)),
        "colorbar": {
            "orientation": "h",
            "ticklabelposition": "inside",
            "tickvals": np.array(list(CellType)) + 0.5,
            "ticktext": ticktext,
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

    def __init__(self, debug=False):
        """Initialize Visualizer object."""
        self._primary = None
        self._graph_metadata = {}
        self._debug = debug

        # https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
        # If we use a dark theme, make the layout background transparent
        themes = [dbc.themes.JOURNAL]

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
            raise ValueError("Added arrays should have the same logger")

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
        # Array annotation for each timestep.
        t_annotations = np.full(t, "", dtype="object")
        # Cell annotation for each cell at every timestep.
        t_cell_annotations = np.full((t, h, w), "", dtype="object")
        for i, timestep in enumerate(timesteps):
            t_arr = timestep[name]
            mask = np.isnan(t_arr["contents"].astype(float))
            t_color_matrix[i][np.nonzero(~mask)] = CellType.FILLED
            t_color_matrix[i][_indices_to_np_indices(
                t_arr[Op.READ])] = CellType.READ
            t_color_matrix[i][_indices_to_np_indices(
                t_arr[Op.WRITE])] = CellType.WRITE
            t_color_matrix[i][_indices_to_np_indices(
                t_arr[Op.MAXMIN])] = CellType.MAXMIN
            t_value_matrix[i] = t_arr["contents"]
            t_annotations[i] = t_arr["annotations"]
            cell_annotations = t_arr["cell_annotations"]
            if cell_annotations:
                t_cell_annotations[i][_indices_to_np_indices(
                    cell_annotations)] = list(cell_annotations.values())

            for write_idx in t_arr[Op.WRITE]:
                indices = (np.s_[i:], *write_idx)
                t_read_matrix[indices] = t_arr[Op.READ]
                t_highlight_matrix[indices] = t_arr[Op.MAXMIN]
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
        t_cell_annotations = metadata["t_cell_annotations"]

        h, w = t_value_matrix.shape[1], t_value_matrix.shape[2]

        # Extra hovertext info:
        # <br>Value: {value_text}<br>Dependencies: {deps_text}
        # (if cell annotation present:) <br>{annotation}
        mask = np.isnan(t_value_matrix.astype(float))
        t_value_matrix[mask] = -99
        value_text = np.where(~mask,
                              t_value_matrix.astype(arr.dtype).astype("str"),
                              "")
        extra_hovertext = np.char.add("<br>Value: ", value_text)

        # Add cell dependencies.
        deps_text = np.where(t_read_matrix == set(), "{}",
                             t_read_matrix.astype("str"))
        extra_hovertext = np.char.add(extra_hovertext, "<br>Dependencies: ")
        extra_hovertext = np.char.add(extra_hovertext, deps_text)

        # Add cell annotations.
        br = np.where(t_cell_annotations == "", "", "<br>")
        extra_hovertext = np.char.add(extra_hovertext, br)
        annotation_hovertext = t_cell_annotations.astype("str")
        extra_hovertext = np.char.add(extra_hovertext, annotation_hovertext)

        # Remove extra info for empty cells.
        extra_hovertext[t_color_matrix == CellType.EMPTY] = ""

        # Create the figure.
        column_alias = dict(enumerate(kwargs["column_labels"]))
        row_alias = dict(enumerate(kwargs["row_labels"]))
        figure = go.Figure(
            layout={
                "title": arr.array_name,
                "title_x": 0.5,
                "height": max(100 * h, 300),
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
                    "zeroline": False,
                    "scaleanchor": "x",
                },
                "coloraxis": {
                    "showscale": False,
                },
                "clickmode": "event+select",
                "hoverlabel": {
                    "namelength": -1,
                },
            })

        hovertemplate = "<b>%{y}, %{x}</b>%{customdata}<extra></extra>"
        if h == 1:
            hovertemplate = "<b>%{x}</b>%{customdata}<extra></extra>"
        if w == 1:
            hovertemplate = "<b>%{y}</b>%{customdata}<extra></extra>"
        for color, val, extra in zip(t_color_matrix, value_text,
                                     extra_hovertext):
            figure.add_heatmap(
                z=color,
                text=val,
                texttemplate="%{text}",
                textfont={"size": 20},
                customdata=extra,
                hovertemplate=hovertemplate,
                **_get_colorbar_kwargs(colorscale_name),
                xgap=1,
                ygap=1,
                visible=False,
                showscale=self._primary == arr.array_name,
                # showscale=False,
            )

        return self._show_figure_trace(figure, 0)

    def _attach_callbacks(self):
        """Attach callbacks."""
        values = self._graph_metadata[self._primary]["t_value_matrix"]
        t_write_matrix = self._graph_metadata[self._primary]["t_write_matrix"]
        t_read_matrix = self._graph_metadata[self._primary]["t_read_matrix"]

        main_figure = self._graph_metadata[self._primary]["figure"]

        output_figure = [
            Output(name, "figure", allow_duplicate=True)
            for name in self._graph_metadata
        ]

        def make_tests(t, selected_tests):
            if self._debug:
                print("[CALLBACK] helper")
            # On the last timestep, turn off self testing.
            if t == len(values) - 1:
                return {"tests": []}

            # Create list of write indices for t+1.
            write_mask = t_write_matrix[t + 1]
            all_writes = list(np.transpose(np.nonzero(write_mask)))

            # Create list of dependencies for t+1.
            # Any all writes have the same reads on the same timestep, so we
            # arbitrarily pick the first one.
            all_reads = list(t_read_matrix[t + 1][write_mask][0])

            # Populate test_q according to what tests are selected.
            test_q = []

            # NOTE: Render: (Index, Color) .
            if "What is the next cell?" in selected_tests:
                # Write test.
                test_q.append({
                    "truth": all_writes,
                    "render": [],
                    "color": CellType.WRITE,
                    "expected_triggered_id": self._primary,
                    "type": TestType.WRITE,
                    "tip":
                        "What cells are written to in the next frame? (Click "
                        "in any order)"
                })

            if "What are its dependencies?" in selected_tests:
                # Read test.
                test_q.append({
                    "truth": all_reads,
                    "render": [(index, CellType.WRITE) for index in all_writes],
                    "color": CellType.READ,
                    "expected_triggered_id": self._primary,
                    "type": TestType.READ,
                    "tip": "What cells are read for the next timestep? (Click "
                           "in any order)"
                })

            if "What is its value?" in selected_tests:
                # Value tests.
                r = [(index, CellType.READ) for index in all_reads]
                for x, y in zip(*np.nonzero(write_mask)):
                    test_q.append({
                        "truth": [values[t + 1][x][y]],
                        # "render": list(all_reads) + [(x, y)],
                        "render": r + [[(x, y), CellType.WRITE]],
                        "color": CellType.WRITE,
                        "expected_triggered_id": "user-input",
                        "type": TestType.VALUE,
                        "tip": f"What is the value of cell ({x}, {y})?"
                    })
            return {"tests": test_q}

        @self.app.callback(output_figure, Input("slider", "value"),
                           State("test-info", "data"))
        def update_figure(t, info):
            """Update each graph based on the slider value."""
            if self._debug:
                print("[CALLBACK] update_figure")
            # Edge case: in self testing mode and ran out of tests.
            if t > len(values):
                return dash.no_update

            next_figures = [
                self._show_figure_trace(metadata["figure"], t)
                for metadata in self._graph_metadata.values()
            ]

            # Slider changed
            if not info["tests"]:
                # Not in self testing mode, update all figures
                return next_figures

            # Case: Finished all tests of previous input.
            # Change slider and then display tests.
            # Change the main figure (first figure in next_figures list).
            # TODO: This is not the right way to do things.
            # Dicts are usually not ordered.
            next_figures[0], _ = display_tests(info, t)

            return next_figures

        @self.app.callback(Output("array-annotation", "children"),
                           Output("array-annotation", "style"),
                           Input("slider", "value"),
                           prevent_initial_call=False)
        def update_annotation(t):
            """Update the annotation based on the slider value."""
            annotation = ""
            for _, metadata in self._graph_metadata.items():
                ann = metadata["t_annotations"][t]
                if not ann:
                    continue
                annotation += ann

            # Hides the textbox if annotation is empty.
            style = {}
            if not annotation:
                style = {"display": "none"}
            return annotation, style

        @self.app.callback(
            Output("slider", "value", allow_duplicate=True),
            Input("store-keypress", "data"),
            Input("interval", "n_intervals"),
            State("slider", "value"),
        )
        def update_slider(key_data, _, t):
            """Update the value of slider based on state of play/stop button."""
            if self._debug:
                print("[CALLBACK] update_slider")
            if ctx.triggered_id == "interval":
                return (t + 1) % len(values)
            if key_data in [37, 39]:
                return (t + key_data - 38) % len(values)
            return dash.no_update

        @self.app.callback(
            Output("interval", "max_intervals"),
            Input("play", "n_clicks"),
            Input("stop", "n_clicks"),
            Input("self-test-button", "n_clicks"),
        )
        def play_pause_playback(_start_clicks, _stop_clicks, _n_clicks):
            """Starts and stop playback from running.

            Pauses the playback when "stop" or "self-test-button" is pressed.
            """
            if self._debug:
                print("[CALLBACK] play_pause_playback")
            if ctx.triggered_id == "play":
                return -1  # Runs interval indefinitely.
            if ctx.triggered_id in ["stop", "self-test-button"]:
                return 0  # Stops interval from running.
            return dash.no_update

        @self.app.callback(
            Output("playback-control", "style"),
            Input("test-info", "data"),
        )
        def toggle_layout(info):
            if self._debug:
                print("[CALLBACK] toggle_layout")
            if info["tests"]:
                return {"visibility": "hidden"}
            return {"visibility": "visible"}

        @self.app.callback(
            Output("test-info", "data", allow_duplicate=True),
            Output("test-mode-toggle", "children"),
            Input("self-test-button", "n_clicks"),
            State("test-info", "data"),
            State("slider", "value"),
            State("test-select-checkbox", "value"),
        )
        def toggle_test_mode(_, info, t, selected_tests):
            """Toggles self-testing mode.

            This callback performs two task:
            1. Populates the test queue according to what tests are selected by
            the checkbox.
            2. Change the style of the self-test-button component.

            This callback is triggered by clicking the self-test-button
            component and updates the test info.
            """
            if self._debug:
                print("[CALLBACK] toggle_test_mode")
            test_button = dbc.Button("Test Myself!",
                                     id="self-test-button",
                                     class_name="h-100",
                                     color="info")
            # No tests to be performed on the last timestep.
            if t == len(values) - 1:
                # TODO: notify user that there is no more testing
                return {"tests": []}, test_button

            # Turn off testing mode if no tests selected or it was already on.
            if info["tests"] or not selected_tests:
                return {"tests": []}, test_button

            test_button = dbc.Button("Exit Testing Mode",
                                     id="self-test-button",
                                     class_name="h-100",
                                     color="warning")

            # Update test-info with selected tests on this timestep.
            return make_tests(t, selected_tests), test_button

        @self.app.callback(
            Output(self._primary, "figure", allow_duplicate=True),
            Output("test-instructions", "children"),
            Input("test-info", "data"),
            State("slider", "value"),
        )
        def display_tests(info, t):
            if self._debug:
                print("[CALLBACK] display_tests")
            alert = dbc.Alert(is_open=False,
                              color="danger",
                              class_name="alert-auto")
            if not info["tests"]:
                return self._show_figure_trace(main_figure, t), alert

            fig = copy.deepcopy(main_figure)

            # Clear HIGHLIGHT, READ, and WRITE cells to FILLED.
            z = fig.data[t].z.astype("bool").astype("int")

            # Highlight the revelant cells as specified by "render".
            test = info["tests"][0]
            render = test["render"]
            for (x, y), color in render:
                z[x][y] = color

            # Bring up test-specific instructions.
            alert.is_open = True
            alert.children = test["tip"]

            return fig.update_traces(z=z, selector=t), alert

        @self.app.callback(
            Output("test-info", "data", allow_duplicate=True),
            Output("correct-alert", "children"),
            # For manually resetting clickData.
            Output(self._primary, "clickData"),
            Output("slider", "value", allow_duplicate=True),
            # Trigger this callback every time "enter" is pressed.
            Input("user-input", "n_submit"),
            Input(self._primary, "clickData"),
            State("user-input", "value"),
            State("test-info", "data"),
            State("slider", "value"),
            State("test-select-checkbox", "value"),
        )
        def validate(_, click_data, user_input, info, t, selected_tests):
            """Validates the user input."""
            if self._debug:
                print("[CALLBACK] validate")
            if not info["tests"]:
                return dash.no_update

            test = info["tests"][0]
            if ctx.triggered_id != test["expected_triggered_id"]:
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

            # Construct alert hint.
            test_type = test["type"]
            alert_hint = ""
            if test_type == TestType.READ:
                alert_hint = ("The selected cell was not read from. Try "
                              "clicking a different cell.")
            elif test_type == TestType.WRITE:
                alert_hint = ("The selected cell was not written to. Try "
                              "clicking on a different cell.")
            elif test_type == TestType.VALUE:
                alert_hint = (f"{answer} is the incorrect value. Try entering "
                              f"another value.")
            else:
                raise ValueError(f"Invalid test type {test_type}")

            # The alert for correct or incorrect input.
            correct_alert = dbc.Alert(
                [
                    html.H4("Incorrect!"),
                    html.Hr(),
                    html.P(alert_hint),
                ],
                color="danger",
                is_open=True,
                dismissable=True,
                class_name="alert-auto",
            )

            # If answer is correct, remove from truth and render the test
            # values. Also updates alert.
            truths = test["truth"]
            if answer in truths:
                truths.remove(answer)
                test["render"].append([answer, test["color"]])

                # Construct alert hint.
                test_type = test["type"]
                if test_type == TestType.READ:
                    alert_hint = ("Continue clicking on cells that were read "
                                  "from.")
                elif test_type == TestType.WRITE:
                    alert_hint = ("Continue clicking on cells that were "
                                  "written to.")
                elif test_type == TestType.VALUE:
                    alert_hint = "Enter the value of the next cell."

                correct_alert.children = [
                    html.H4("Correct!"),
                    html.Hr(),
                    html.P(alert_hint)
                ]
                correct_alert.color = "success"

            # If all truths have been found, pop from test queue.
            if not truths:
                info["tests"].pop(0)

                # If all tests are done, update slider value and make tests.
                if not info["tests"]:
                    new_info = make_tests(t + 1, selected_tests)

                    # Hint: starting new tests for the next timestep or testing
                    #       mode terminated.
                    alert_hint = ("You have completed all tests for this "
                                  "timestep.")
                    if new_info["tests"]:
                        next_test = new_info["tests"][0]["type"]
                        alert_hint += (f"Starting {TestType(next_test).name}"
                                       f" test for the next timestep.")
                    else:
                        alert_hint += "There are no more tests available."

                    correct_alert.children[2] = html.P(alert_hint)
                    return new_info, correct_alert, None, t + 1

                # Hint: starting new tests for the same timestep.
                next_test = info["tests"][0]["type"]
                alert_hint = (f"{TestType(test_type).name} test complete. You "
                              f"are moving on to the "
                              f"{TestType(next_test).name} test.")
                correct_alert.children[2] = html.P(alert_hint)

            # Updates test info, the alert, and resets clickData.
            return info, correct_alert, None, dash.no_update

        @self.app.callback(
            Output(self._primary, "figure", allow_duplicate=True),
            Input(self._primary, "clickData"),
            State("test-info", "data"),
            State("slider", "value"),
        )
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
            z[_indices_to_np_indices(h[t][y][x])] = CellType.MAXMIN

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
            dbc.Col(dbc.Button("Test Myself!",
                               id="self-test-button",
                               class_name="h-100",
                               color="info"),
                    width="auto",
                    id="test-mode-toggle"),
            dbc.Col(
                dbc.Checklist(questions, questions, id="test-select-checkbox"))
        ])

        description_md = [
            dcc.Markdown(metadata["description"], mathjax=True)
            for metadata in self._graph_metadata.values()
        ]

        sidebar = html.Div([
            dbc.Stack(
                [
                    *description_md,
                    test_select_checkbox,
                    # User input box.
                    dbc.Input(id="user-input",
                              type="number",
                              placeholder="Enter value here",
                              className="my-1"),
                    # Textbox to display array annotations.
                    dcc.Markdown(
                        "",
                        id="array-annotation",
                        className=("bg-secondary-subtle text-center py-3"
                                   "rounded"),
                        style={"display": "none"}),
                    # An alert to display the test instructions.
                    html.Div(id="test-instructions", className="mx-3"),
                    # An alert to display the correctness of the input.
                    html.Div(id="correct-alert", className="mx-3"),
                ],
                id="sidebar",
                className="bg-secondary vh-100 px-3"),
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
                            class_name="g-1",
                            align="center",
                        ),
                        dbc.Row(
                            dbc.Stack(graphs),
                            id="page-content",
                            align="center",
                        ),
                    ],
                            width=8),
                ]),
                *datastores,
            ],
            fluid=True,
        )

        self._attach_callbacks()

        self.app.run_server(debug=not self._debug, use_reloader=True)

    @property
    def app(self):
        """Returns the Dash app object."""
        return self._app
