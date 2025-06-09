import mkdocs_gen_files

demos = {
    "knapsack": "https://knapsack.demos.davidhaolong.com",
    "strange_printer": "https://strange-printer.demos.davidhaolong.com",
    "weighted_interval_scheduling": "https://wis.demos.davidhaolong.com",
}

index_lines = [
    "# Live Demos\n", "Try out these interactive examples:\n"
]

for slug, url in demos.items():
    title = slug.replace("_", " ").title()
    content = f"""
# {title}

<iframe src="{url}"
        width="100%"
        height="600"
        style="border:1px solid #ccc; border-radius: 8px;">
</iframe>
"""
    path = f"demos/{slug}.md"
    with mkdocs_gen_files.open(path, "w") as f:
        print(content, file=f)
        print(f"Generated {path}")

    mkdocs_gen_files.set_edit_path(path, "docs/scripts/gen_demo_pages.py")

    index_lines.append(f"- [{title}]({slug}.md)")

with mkdocs_gen_files.open("demos/index.md", "w") as f:
    f.write("\n".join(index_lines))
