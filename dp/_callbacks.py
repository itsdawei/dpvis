"""This file provides the callbacks attached to the visualizer."""
import copy
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html
from enum import IntEnum

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
    """TestType is used to distinguish between tests in test-info."""
    READ = 0
    WRITE = 1
    VALUE = 2


def make_tests(visualizer, t, selected_tests):
    values = (visualizer._graph_metadata[visualizer._primary]["t_value_matrix"])
    t_write_matrix = (
        visualizer._graph_metadata[visualizer._primary]["t_write_matrix"])
    t_read_matrix = (
        visualizer._graph_metadata[visualizer._primary]["t_read_matrix"])

    if visualizer._debug:
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
            "expected_triggered_id": visualizer._primary,
            "type": TestType.WRITE,
            "tip": "What cells are written to in the next frame? (Click "
                   "in any order)"
        })

    if "What are its dependencies?" in selected_tests:
        # Read test.
        test_q.append({
            "truth": all_reads,
            "render": [(index, CellType.WRITE) for index in all_writes],
            "color": CellType.READ,
            "expected_triggered_id": visualizer._primary,
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


def attach_slider_updates(visualizer):
    values = (visualizer._graph_metadata[visualizer._primary]["t_value_matrix"])
    t_write_matrix = (
        visualizer._graph_metadata[visualizer._primary]["t_write_matrix"])
    t_read_matrix = (
        visualizer._graph_metadata[visualizer._primary]["t_read_matrix"])
    main_figure = visualizer._graph_metadata[visualizer._primary]["figure"]

    output_figure = [
        Output(name, "figure", allow_duplicate=True)
        for name in visualizer._graph_metadata
    ]

    @visualizer.app.callback(
        output_figure,
        Input("slider", "value"),
        State("test-info", "data"),
    )
    def update_figure(t, info):
        """Update each graph based on the slider value."""
        if visualizer._debug:
            print("[CALLBACK] update_figure")

        # Edge case: in self testing mode and ran out of tests.
        if t > len(values):
            return dash.no_update

        next_figures = [
            visualizer._show_figure_trace(metadata["figure"], t)
            for metadata in visualizer._graph_metadata.values()
        ]

        # Slider changed
        if not info["tests"]:
            # Not in self testing mode, update all figures
            return next_figures

        # Case: Finished all tests of previous input.
        # Change slider and then display tests.
        # Change the main figure (first figure in next_figures list).
        # TODO: THIS IS HOT FIX FOR BUG X
        next_figures[0], _ = display_tests(info, t)

        return next_figures

    # TODO: REMOVE THIS FUNCTION AFTER FIXING BUG X.
    def display_tests(info, t):
        if visualizer._debug:
            print("[CALLBACK] display_tests")
        alert = dbc.Alert(is_open=False,
                          color="danger",
                          class_name="alert-auto")
        if not info["tests"]:
            return visualizer._show_figure_trace(main_figure, t), alert

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

    @visualizer.app.callback(
        Output("array-annotation", "children"),
        Output("array-annotation", "style"),
        Input("slider", "value"),
        prevent_initial_call=False,
    )
    def update_annotation(t):
        """Update the annotation based on the slider value."""
        annotation = ""
        for name, metadata in visualizer._graph_metadata.items():
            ann = metadata["t_annotations"][t]
            if not ann:
                continue
            annotation += f"[{name}] {ann}"

        # Hides the textbox if annotation is empty.
        style = {}
        if not annotation:
            style = {"display": "none"}
        return annotation, style

    @visualizer.app.callback(
        Output("slider", "value", allow_duplicate=True),
        Input("store-keypress", "data"),
        Input("interval", "n_intervals"),
        State("slider", "value"),
    )
    def update_slider(key_data, _, t):
        """Update the value of slider based on state of play/stop button."""
        if visualizer._debug:
            print("[CALLBACK] update_slider")
        if ctx.triggered_id == "interval":
            return (t + 1) % len(values)
        if key_data in [37, 39]:
            return (t + key_data - 38) % len(values)
        return dash.no_update


def attach_test_mode(visualizer):
    values = (visualizer._graph_metadata[visualizer._primary]["t_value_matrix"])
    t_write_matrix = (
        visualizer._graph_metadata[visualizer._primary]["t_write_matrix"])
    t_read_matrix = (
        visualizer._graph_metadata[visualizer._primary]["t_read_matrix"])
    main_figure = visualizer._graph_metadata[visualizer._primary]["figure"]

    @visualizer.app.callback(
        Output("interval", "max_intervals"),
        Input("play", "n_clicks"),
        Input("stop", "n_clicks"),
        Input("self-test-button", "n_clicks"),
    )
    def play_pause_playback(_start_clicks, _stop_clicks, _n_clicks):
        """Starts and stop playback from running.

        Pauses the playback when "stop" or "self-test-button" is pressed.
        """
        if visualizer._debug:
            print("[CALLBACK] play_pause_playback")
        if ctx.triggered_id == "play":
            return -1  # Runs interval indefinitely.
        if ctx.triggered_id in ["stop", "self-test-button"]:
            return 0  # Stops interval from running.
        return dash.no_update

    @visualizer.app.callback(
        Output("playback-control", "style"),
        Input("test-info", "data"),
    )
    def toggle_layout(info):
        if visualizer._debug:
            print("[CALLBACK] toggle_layout")
        if info["tests"]:
            return {"visibility": "hidden"}
        return {"visibility": "visible"}

    @visualizer.app.callback(
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
        if visualizer._debug:
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
        return make_tests(visualizer, t, selected_tests), test_button

    @visualizer.app.callback(
        Output(visualizer._primary, "figure", allow_duplicate=True),
        Output("test-instructions", "children"),
        Input("test-info", "data"),
        State("slider", "value"),
    )
    def display_tests(info, t):
        if visualizer._debug:
            print("[CALLBACK] display_tests")
        alert = dbc.Alert(is_open=False,
                          color="danger",
                          class_name="alert-auto")
        if not info["tests"]:
            return visualizer._show_figure_trace(main_figure, t), alert

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

    @visualizer.app.callback(
        Output("test-info", "data", allow_duplicate=True),
        Output("correct-alert", "children"),
        # For manually resetting clickData.
        Output(visualizer._primary, "clickData"),
        Output("slider", "value", allow_duplicate=True),
        # Trigger this callback every time "enter" is pressed.
        Input("user-input", "n_submit"),
        Input(visualizer._primary, "clickData"),
        State("user-input", "value"),
        State("test-info", "data"),
        State("slider", "value"),
        State("test-select-checkbox", "value"),
    )
    def validate(_, click_data, user_input, info, t, selected_tests):
        """Validates the user input."""
        if visualizer._debug:
            print("[CALLBACK] validate")
        if not info["tests"]:
            return dash.no_update

        test = info["tests"][0]
        if ctx.triggered_id != test["expected_triggered_id"]:
            return dash.no_update

        if ctx.triggered_id == visualizer._primary:
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
                html.Hr(), html.P(alert_hint)
            ]
            correct_alert.color = "success"

        # If all truths have been found, pop from test queue.
        if not truths:
            info["tests"].pop(0)

            # If all tests are done, update slider value and make tests.
            if not info["tests"]:
                new_info = make_tests(visualizer, t + 1, selected_tests)

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


def attach_dependencies(visualizer):

    @visualizer.app.callback(
        Output(visualizer._primary, "figure", allow_duplicate=True),
        Input(visualizer._primary, "clickData"),
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
        d = visualizer._graph_metadata[visualizer._primary]["t_read_matrix"]
        z[_indices_to_np_indices(d[t][y][x])] = CellType.READ

        # Highlight highlights.
        h = visualizer._graph_metadata[
            visualizer._primary]["t_highlight_matrix"]
        z[_indices_to_np_indices(h[t][y][x])] = CellType.MAXMIN

        return fig.update_traces(z=z, selector=t)
