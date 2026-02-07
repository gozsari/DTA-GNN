#!/usr/bin/env python3
"""Generate a PDF from all documentation markdown files."""

import os
import sys
from pathlib import Path
from typing import List, Any


def parse_nav(nav_list: List[Any], docs_dir: Path, files: List[Path]) -> None:
    """Recursively parse navigation structure to collect markdown files in order."""
    for item in nav_list:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str):
                    # Single file entry: { "Title": "path/to/file.md" }
                    file_path = docs_dir / value
                    if file_path.exists() and file_path not in files:
                        files.append(file_path)
                elif isinstance(value, list):
                    # Nested section: { "Section": [ ... ] }
                    parse_nav(value, docs_dir, files)
        elif isinstance(item, str):
            # Direct file reference
            file_path = docs_dir / item
            if file_path.exists() and file_path not in files:
                files.append(file_path)


def combine_markdown_files(files: List[Path], output_path: Path) -> None:
    """Combine multiple markdown files into one."""
    with open(output_path, "w", encoding="utf-8") as outfile:
        for i, file_path in enumerate(files):
            # Add a page break before each section (except the first)
            if i > 0:
                outfile.write("\n\n\\newpage\n\n")

            # Read and write file content
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()

                # Remove HTML divs and other HTML-only features that don't work in PDF
                # This is a basic cleanup - you might need more sophisticated processing
                content = content.replace('<div align="center">', "")
                content = content.replace("</div>", "")

                # Remove image references (they won't work in PDF without paths)
                # Or we could handle them differently
                lines = content.split("\n")
                cleaned_lines = []
                for line in lines:
                    if line.strip().startswith("<img") and "assets/logo.png" in line:
                        # Skip logo image
                        continue
                    if line.strip().startswith("<p>") and line.strip().endswith("</p>"):
                        # Convert simple <p> tags to markdown
                        cleaned_lines.append(
                            line.replace("<p>", "")
                            .replace("</p>", "")
                            .replace("<strong>", "**")
                            .replace("</strong>", "**")
                        )
                    else:
                        cleaned_lines.append(line)

                content = "\n".join(cleaned_lines)

                outfile.write(content)
                outfile.write("\n\n")


def get_nav_order() -> List[Any]:
    """Return navigation order from mkdocs.yml structure."""
    # Manually define the nav structure based on mkdocs.yml
    return [
        "index.md",
        {
            "Getting Started": [
                "getting-started/installation.md",
                "getting-started/quickstart.md",
            ]
        },
        {
            "User Guide": [
                "user-guide/data-sources.md",
                "user-guide/cleaning.md",
                "user-guide/splits.md",
                "user-guide/audits.md",
            ]
        },
        {
            "Interfaces": [
                "interfaces/cli.md",
                "interfaces/ui.md",
                "interfaces/python-api.md",
            ]
        },
        {"Modeling": ["modeling/features.md", "modeling/models.md"]},
        {"Hyperparameter Optimization": ["hpo/hyperopt.md"]},
        {"Development": ["development/contributing.md"]},
    ]


def main():
    """Main function to generate PDF from documentation."""
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"

    # Get navigation order
    nav = get_nav_order()

    # Collect all markdown files in navigation order
    files: List[Path] = []
    parse_nav(nav, docs_dir, files)

    if not files:
        print("Error: No markdown files found in navigation")
        sys.exit(1)

    print(f"Found {len(files)} documentation files:")
    for f in files:
        print(f"  - {f.relative_to(project_root)}")

    # Create combined markdown file
    combined_md = project_root / "docs_combined.md"
    combine_markdown_files(files, combined_md)
    print(f"\nCombined markdown written to: {combined_md}")

    # Convert to PDF using pandoc
    output_pdf = project_root / "DTA-GNN-Documentation.pdf"

    print(f"\nConverting to PDF: {output_pdf}")

    # Pandoc command with good PDF settings
    pandoc_cmd = (
        f"pandoc {combined_md} "
        f"-o {output_pdf} "
        f"--pdf-engine=xelatex "
        f"-V geometry:margin=1in "
        f"-V fontsize=11pt "
        f"-V documentclass=article "
        f"--toc "
        f"--toc-depth=3 "
        f"--highlight-style=tango "
        f"--metadata title='DTA-GNN Documentation' "
        f"--metadata author='DTA-GNN Team' "
        f"-N "  # Number sections
    )

    # Try xelatex first, fall back to pdflatex
    exit_code = os.system(pandoc_cmd)

    if exit_code != 0:
        print("\nWarning: xelatex failed, trying pdflatex...")
        pandoc_cmd_pdflatex = (
            f"pandoc {combined_md} "
            f"-o {output_pdf} "
            f"--pdf-engine=pdflatex "
            f"-V geometry:margin=1in "
            f"-V fontsize=11pt "
            f"-V documentclass=article "
            f"--toc "
            f"--toc-depth=3 "
            f"--highlight-style=tango "
            f"--metadata title='DTA-GNN Documentation' "
            f"--metadata author='DTA-GNN Team' "
            f"-N "
        )
        exit_code = os.system(pandoc_cmd_pdflatex)

    if exit_code == 0:
        print(f"\n✓ PDF generated successfully: {output_pdf}")
        # Clean up combined markdown (optional)
        # combined_md.unlink()
    else:
        print(f"\n✗ Error generating PDF. Exit code: {exit_code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
