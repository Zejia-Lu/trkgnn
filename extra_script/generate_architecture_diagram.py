#!/usr/bin/env python3
"""
Generate Graphviz DOT diagrams for LinkNet and Enhanced_TrackNet_final.

Example:
  python extra_script/generate_architecture_diagram.py --model linknet --out linknet.dot --render
  python extra_script/generate_architecture_diagram.py --model enhanced_track_net_final --out enhanced.dot --render
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _dot_linknet() -> str:
    return r"""digraph LinkNet {
  rankdir=TB;
  graph [fontsize=12, fontname="Helvetica", labelloc=t, label="LinkNet"];
  node [shape=box, style="rounded,filled", fillcolor="#e7f4ff", fontname="Helvetica"];
  edge [color="#444444", arrowsize=0.8];

  input_node [label="Node Features\n(x, y, z, Bx, By, Bz)"];
  input_edge [label="Edge Attributes\n(R, theta, phi)"];

  node_embed [label="Node Embedding\n(MLP)"];
  edge_embed [label="Edge Embedding\n(MLP)"];

  combined [label="Combined Feature\n(Scatter Add + Concat)"];
  transconv [label="TransformerConv\n(iterate N times)", fillcolor="#ffd6d6"];
  edge_clf [label="Edge Classifier\n(MLP)", fillcolor="#ffd6d6"];
  edge_out [label="Edge Scores (y)"];

  input_node -> node_embed;
  input_edge -> edge_embed;
  node_embed -> combined;
  edge_embed -> combined;
  combined -> transconv;
  transconv -> edge_clf;
  edge_clf -> edge_out;
}
"""


def _dot_enhanced_track_net_final() -> str:
    return r"""digraph EnhancedTrackNetFinal {
  rankdir=TB;
  graph [fontsize=12, fontname="Helvetica", labelloc=t, label="TrackNet"];
  node [shape=box, style="rounded,filled", fillcolor="#e7f4ff", fontname="Helvetica"];
  edge [color="#444444", arrowsize=0.8];

  input_node [label="Node Features\n(x, y, z, Bx, By, Bz)"];
  input_edge [label="Edge Attributes\n(R, theta, phi, dx, dy, dz)"];

  z_embed [label="Z Embedding\n(learned)", fillcolor="#fff0cc"];

  node_embed [label="Node Embedding\n(MLP)"];
  node_cbam [label="CBAM (Node)", fillcolor="#ffd6d6"];

  edge_embed [label="Edge Embedding\n(MLP)"];
  edge_cbam [label="CBAM (Edge)", fillcolor="#ffd6d6"];

  combined [label="Combined Feature\n(Scatter Add + Concat)"];
  transconv [label="TransformerConv\n(iterate N times)", fillcolor="#ffd6d6"];
  gfm [label="Gated Fusion\n(update node/edge)", fillcolor="#ffd6d6"];

  edge_clf_in [label="Concat Node Pair + Edge Attr"];
  edge_clf [label="Edge Classifier\n(MLP)", fillcolor="#ffd6d6"];
  edge_out [label="Edge Scores (y)"];

  input_node -> node_embed;
  input_node -> z_embed;
  node_embed -> node_cbam;
  z_embed -> node_cbam;
  node_cbam -> combined;

  input_edge -> edge_embed;
  edge_embed -> edge_cbam;
  edge_cbam -> combined;

  combined -> transconv;
  transconv -> gfm;
  gfm -> combined [style=dashed, label="iterate"];

  gfm -> edge_clf_in;
  input_edge -> edge_clf_in;
  edge_clf_in -> edge_clf;
  edge_clf -> edge_out;
}
"""


def _write_out(path: Path, dot: str) -> None:
    path.write_text(dot, encoding="utf-8")


def _render_dot(dot_path: Path, fmt: str) -> None:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        raise RuntimeError("graphviz 'dot' not found in PATH")
    out_path = dot_path.with_suffix(f".{fmt}")
    cmd = [dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out_path)]
    result = shutil.which("dot")
    if not result:
        raise RuntimeError("graphviz 'dot' not found in PATH")
    import subprocess

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture diagram (Graphviz DOT).")
    parser.add_argument(
        "--model",
        required=True,
        choices=["linknet", "enhanced_track_net_final"],
        help="Model diagram to generate.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output DOT file path.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with graphviz 'dot' if available.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Render format (png, svg, pdf).",
    )
    args = parser.parse_args()

    if args.model == "linknet":
        dot = _dot_linknet()
    else:
        dot = _dot_enhanced_track_net_final()

    out_path = Path(args.out)
    _write_out(out_path, dot)

    if args.render:
        _render_dot(out_path, args.format)


if __name__ == "__main__":
    main()
