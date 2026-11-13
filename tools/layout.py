#!/usr/bin/env python3
"""Write neuronte-gpu layout files from a Braille experiment manifest.

A layout names the networks of one model set and the pixel rectangle each
covers in a cell of a given size:

    neuronte-gpu-layout 1
    name parts24-s0
    cell 16 32
    span 0x2800 0x28ff          (bundles only: the scalars the member covers)
    tile MASK X Y W H FILE      (one per network; MASK -1 passes the scalar through)

A "parts" case becomes one tile per region: the region's x domain [x0, x1]
(fractions of the cell width) and y domain [y0, y1] (on the [0, 2] grid) map
to pixels x0*W .. x1*W and y0*H/2 .. y1*H/2, which must be whole numbers.
Every tile is evaluated on its own local [0, 1] x [0, 2] grid with the scalar
masked to the dots it draws, exactly as the reference worker calls
neuronte_coverage per tile. A "bundle" case is one tile over the whole cell
naming the bundle file; the library resolves the member that the range table
routes the span to, and recomputes scalars outside the span on the CPU.

usage:
    layout.py MANIFEST CASE WxH [-o FILE]
    layout.py MANIFEST --all DIR [--cells 8x16,16x32,48x96]
"""

import argparse
import json
import sys
from pathlib import Path

BRAILLE_SPAN = (0x2800, 0x28FF)


def cell(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def layout_text(manifest, case_name, cell_w, cell_h):
    case = next((c for c in manifest["cases"] if c["name"] == case_name), None)
    if case is None:
        raise ValueError(f"no case {case_name!r} in the manifest")
    models = manifest["models"]
    lines = ["neuronte-gpu-layout 1", f"name {case_name}", f"cell {cell_w} {cell_h}"]
    if case.get("kind") != "parts":
        if len(case["members"]) != 1:
            raise ValueError(f"{case_name}: a bundle case must have one member")
        lines.append("span 0x%04x 0x%04x" % BRAILLE_SPAN)
        lines.append(f"tile -1 0 0 {cell_w} {cell_h} {models[case['members'][0]]['file']}")
        return "\n".join(lines) + "\n"
    for member, region in zip(case["members"], case["regions"], strict=True):
        x0, x1 = region["x_domain"]
        y0, y1 = region["y_domain"]
        edges = (x0 * cell_w, x1 * cell_w, y0 * cell_h / 2, y1 * cell_h / 2)
        if any(int(v) != v for v in edges):
            raise ValueError(f"{case_name}: a {cell_w}x{cell_h} cell does not divide evenly into the layout")
        x, xe, y, ye = map(int, edges)
        lines.append(f"tile {int(region['mask'])} {x} {y} {xe - x} {ye - y} {models[member]['file']}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("case", nargs="?")
    ap.add_argument("cell", nargs="?", type=cell)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--all", type=Path, metavar="DIR", help="write CASE-WxH.layout for every case and cell")
    ap.add_argument("--cells", default="8x16,16x32,48x96")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if a.all:
        a.all.mkdir(parents=True, exist_ok=True)
        for c in manifest["cases"]:
            for w, h in map(cell, a.cells.split(",")):
                try:
                    text = layout_text(manifest, c["name"], w, h)
                except ValueError as e:
                    print(f"skipped: {e}", file=sys.stderr)
                    continue
                (a.all / f"{c['name']}-{w}x{h}.layout").write_text(text)
        return 0
    if not a.case or not a.cell:
        ap.error("CASE and WxH are required without --all")
    text = layout_text(manifest, a.case, *a.cell)
    if a.output:
        a.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
