#!/usr/bin/env python3
"""Convert a Jupyter notebook to PDF."""

import os
import sys
import tempfile
from pathlib import Path


def convert_notebook_to_html(notebook_path: Path, html_path: Path) -> None:
    """Convert notebook to HTML using nbconvert Python API."""
    try:
        import nbformat
        from nbconvert import HTMLExporter
    except ImportError:
        print("\n✗ Error: nbconvert not installed")
        print("Please install: pip install nbconvert")
        sys.exit(1)
    
    print(f"Converting notebook to HTML using nbconvert API...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Export to HTML
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'  # Use classic template
    
    try:
        (body, resources) = html_exporter.from_notebook_node(notebook)
    except Exception as e:
        # Try without specifying template
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(notebook)
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"✓ HTML generated: {html_path}")


def convert_html_to_pdf_with_playwright(html_path: Path, pdf_path: Path) -> None:
    """Convert HTML file to PDF using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("\n✗ Error: playwright not installed")
        print("Please install: pip install playwright && playwright install chromium")
        sys.exit(1)
    
    print(f"Converting HTML to PDF using Playwright...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Load the HTML file
        html_url = f"file://{html_path.resolve()}"
        page.goto(html_url, wait_until="networkidle")
        
        # Wait a bit for any dynamic content
        page.wait_for_timeout(1000)
        
        # Generate PDF
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={
                "top": "0.75in",
                "right": "0.75in",
                "bottom": "0.75in",
                "left": "0.75in",
            }
        )
        
        browser.close()
    
    print(f"✓ PDF generated: {pdf_path}")


def convert_notebook_to_pdf(
    notebook_path: Path, 
    output_path: Path = None, 
    method: str = "html"
) -> None:
    """
    Convert a Jupyter notebook to PDF.
    
    Args:
        notebook_path: Path to the notebook file
        output_path: Optional output PDF path. If not provided, uses notebook name with .pdf extension
        method: Conversion method - "html" (default, uses HTML+Playwright) or "pdf" (requires LaTeX)
    """
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    if output_path is None:
        output_path = notebook_path.with_suffix('.pdf')
    
    print(f"Converting notebook to PDF...")
    print(f"  Input:  {notebook_path}")
    print(f"  Output: {output_path}")
    
    if method == "html":
        # Step 1: Convert notebook to HTML
        html_path = notebook_path.with_suffix('.html')
        print(f"\nStep 1: Converting notebook to HTML...")
        
        try:
            convert_notebook_to_html(notebook_path, html_path)
            
            # Step 2: Convert HTML to PDF using Playwright
            print(f"\nStep 2: Converting HTML to PDF...")
            convert_html_to_pdf_with_playwright(html_path, output_path)
            
            print(f"\n✓ PDF generated successfully: {output_path}")
            
            # Optionally clean up HTML file
            # html_path.unlink()
            
        except Exception as e:
            print(f"\n✗ Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif method == "pdf":
        # Direct PDF conversion using LaTeX (requires LaTeX installation)
        print(f"\nConverting using LaTeX-based PDF (requires LaTeX installation)...")
        
        try:
            import nbformat
            from nbconvert import PDFExporter
        except ImportError:
            print("\n✗ Error: nbconvert not installed")
            print("Please install: pip install nbconvert")
            sys.exit(1)
        
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Export to PDF
        pdf_exporter = PDFExporter()
        (body, resources) = pdf_exporter.from_notebook_node(notebook)
        
        # Write PDF file
        with open(output_path, 'wb') as f:
            f.write(body)
        
        print(f"\n✓ PDF generated successfully: {output_path}")
    
    else:
        print(f"\n✗ Unknown method: {method}")
        sys.exit(1)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert a Jupyter notebook to PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use HTML method (default, no LaTeX required)
  python convert_notebook_to_pdf.py examples/01_baseline_models.ipynb
  
  # Use LaTeX-based PDF (requires LaTeX installation)
  python convert_notebook_to_pdf.py examples/01_baseline_models.ipynb --method pdf
  
  # Specify output path
  python convert_notebook_to_pdf.py examples/01_baseline_models.ipynb output.pdf
        """
    )
    parser.add_argument("notebook_path", help="Path to the notebook file")
    parser.add_argument("output_path", nargs="?", help="Output PDF path (optional)")
    parser.add_argument(
        "--method",
        choices=["html", "pdf"],
        default="html",
        help="Conversion method: html (default, uses HTML+Playwright, no LaTeX) or pdf (requires LaTeX)"
    )
    
    args = parser.parse_args()
    
    notebook_path = Path(args.notebook_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else None
    
    convert_notebook_to_pdf(notebook_path, output_path, method=args.method)


if __name__ == "__main__":
    main()
