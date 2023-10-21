# HISTORY

## 0.1.0

### API

- Remove "null" text in empty cells (#21)
- Make heatmap colorbar a legend for the colors (#21)
- Enable plotly builtin colorscales for visualizer (#21)
- Implement highlighting for visualizer (#19)
- Enable writing with slices (#18)
- Implement ``visualize`` module for 1D and 2D arrays (#16)
- Raises warning when accessing array out-of-bounds (#15)
- Enable reading with slices (#14)
- Implement method to print timesteps of the logger (#13)
- Add max/min method in ``DPArray``(#12, #17)
- Implement method to convert ``Logger`` object to a timestep action on the
  array (#11)
- Add ``Value`` field to Logger class for tracking (#10)
- Documentation with mkdocs (#7)
- Add Logger to DPArray class (#4)
- Create ``Logger`` class (#4)
- Remove integer support in ``DPArray``(#6)
- Create ``DPArray`` class (#3)
- Initial project setup (#1, #2)

### Documentation
- Excavation and Matrix Traversal examples (#19, #21)

### Improvements
- Migrate to poetry (#21)
- Setup automated test-runners (#5)
    - Reduce GitHub Action usage (#9)
    - mkdocs (#7)
    - pytest (#5)
