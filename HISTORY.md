# HISTORY

## 0.1.0

### API

- Fixing colors in test mode tutorial to account for max/min (#65)
- Adding a test mode tutorial (#64)
- Adding more descriptive alerts in test mode (#57)
- Fix hover-text for 1D arrays (#58)
- Allow `int` dtype for `DPArray` (#53)
- Self-testing mode perists over multiple timesteps (#43)
- Add annotation functionality (#36, #50)
- Add to test queue based on selected tests checklist (#41)
- Combine "Correct!" and "Incorrect!" alert (#40)
- Execute self-tests sequentially (#39)
- Add sidebar in UI (#33)
- Show extra description of the DP (#33)
- Make `create_figure` return a `Figure` object (#34, #35)
- Add support for displaying multiple arrays (#23, #32)
- Improve Front-end Callback Structure (#31, #37)
- Implement object-oriented API (#28)
- Implemented self testing of values (#22)
- Traceback solution verification and visualization (#24)
- Display dependencies on click (#25, #26)
- Migrate from graph objects to Dash (#20)
- Enable arrow keys for time travel (#20)
- Remove "null" text in empty cells (#21)
- Make heatmap colorbar a legend for the colors (#21)
- Enable plotly builtin colorscales for visualizer (#21)
- Implement highlighting for visualizer (#19)
- Enable writing with slices (#18)
- Implement `visualize` module for 1D and 2D arrays (#16)
- Raises warning when accessing array out-of-bounds (#15)
- Enable reading with slices (#14)
- Implement method to print timesteps of the logger (#13)
- Add max/min method in `DPArray`(#12, #17, #44)
- Implement method to convert `Logger` object to a timestep action on the
  array (#11)
- Add `Value` field to Logger class for tracking (#10)
- Add Logger to DPArray class (#4)
- Create `Logger` class (#4)
- Remove integer support in `DPArray`(#6)
- Create `DPArray` class (#3)
- Initial project setup (#1, #2)

### Style

- Remove colorbars in auxiliary heatmaps (#48) 
- `HIGHLIGHT` -> `MAXMIN` (#47)
- Toggle style of test mode button (#45)

### Documentation

- Update installation instruction (#54, #55)
- Add WIS (#53)
- Add "tutorials" and "examples" to documentation (#52)
- Host documentation on readthedocs (#51)
- Excavation and Matrix Traversal examples (#19, #21)
- Documentation with mkdocs (#7)

### Improvements

- Migrate to poetry (#21)
- Setup automated test-runners (#5)
  - Reduce GitHub Action usage (#9)
  - mkdocs (#7)
  - pytest (#5)
