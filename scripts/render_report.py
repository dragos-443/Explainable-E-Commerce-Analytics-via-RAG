"""Render the final Markdown report as a standalone printable HTML document."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from markdown_it import MarkdownIt


STYLE = """
@page { size: A4; margin: 18mm 16mm 20mm; }
:root { color-scheme: light; }
body {
  color: #18212b;
  background: #fff;
  font: 10.5pt/1.5 "Segoe UI", Arial, sans-serif;
  max-width: 190mm;
  margin: 0 auto;
}
h1, h2, h3 { color: #102a43; page-break-after: avoid; }
h1 { font-size: 25pt; border-bottom: 2px solid #2f80ed; padding-bottom: 8px; }
h2 { font-size: 18pt; margin-top: 28px; border-bottom: 1px solid #bcccdc; }
h3 { font-size: 13pt; margin-top: 20px; }
p, li { orphans: 3; widows: 3; }
pre, blockquote, table, img { break-inside: avoid; }
pre {
  white-space: pre-wrap;
  background: #f4f7fa;
  border: 1px solid #d9e2ec;
  border-radius: 5px;
  padding: 10px;
  font-size: 8.5pt;
}
code { font-family: Consolas, "Courier New", monospace; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt; }
th, td { border: 1px solid #bcccdc; padding: 6px 7px; vertical-align: top; }
th { background: #eaf2fb; text-align: left; }
blockquote { border-left: 4px solid #2f80ed; margin-left: 0; padding-left: 14px; color: #334e68; }
img { display: block; max-width: 100%; max-height: 220mm; margin: 14px auto; }
a { color: #1769aa; text-decoration: none; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-uri", required=True)
    parser.add_argument("--title", default="Explainable E-Commerce Analytics via RAG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.read_text(encoding="utf-8")
    markdown = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = markdown.render(source)
    document = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="{html.escape(args.base_uri, quote=True)}">
  <title>{html.escape(args.title)}</title>
  <style>{STYLE}</style>
</head>
<body>{body}</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
