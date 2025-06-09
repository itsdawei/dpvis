from pathlib import Path
import mkdocs_gen_files

for path in sorted(Path("dp").rglob("*.py")):
    module_path = path.relative_to("dp").with_suffix("")
    doc_path = Path("api", module_path.with_suffix(".md"))

    parts = list(module_path.parts)

    if parts[-1] in {"__init__",  "__main__"}:
        continue

    identifier = ".".join(["dp"] + parts)
    title = parts[-1].replace("_", " ").title()  # e.g., _visualizer → Visualizer
    with mkdocs_gen_files.open(doc_path, "w") as f:
        print(f"# {title}\n\n", file=f)
        print(f"::: {identifier}\n", file=f)

    mkdocs_gen_files.set_edit_path(doc_path, path)
