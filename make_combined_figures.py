"""Combine the seven single figures into three multi-panel figures.

  fig_combined_1 = fig1 (hysteresis) + fig2 (phase diagram)
  fig_combined_2 = fig3 (time series) + fig4 (finite-size scaling)
  fig_combined_3 = fig5a (topology) + fig5b (cascade hysteresis) + fig6 (critical mass)

Panels are labelled (a)(b)(c) in the top-left corner. Output at the same
directory as the inputs; single figures are kept untouched.
"""

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

COMBOS = [
    (["fig1_hysteresis.png", "fig2_phase_diagram.png"],
     ["(a)", "(b)"],
     "fig_combined_1.png"),
    (["fig3_time_series.png", "fig4_finite_size.png"],
     ["(a)", "(b)"],
     "fig_combined_2.png"),
    (["fig5a_topology.png", "fig5b_hysteresis.png", "fig6_critical_mass.png"],
     ["(a)", "(b)", "(c)"],
     "fig_combined_3.png"),
]

TARGET_H = 900
GAP = 18
PAD = 24  # white padding around each panel


def get_font(size):
    for name in ("arial.ttf", "Arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    for paths, labels, out_name in COMBOS:
        panels = []
        for p in paths:
            im = Image.open(os.path.join(FIG, p)).convert("RGB")
            w, h = im.size
            nw = int(w * TARGET_H / h)
            im = im.resize((nw, TARGET_H), Image.LANCZOS)
            # pad with white border
            canvas_p = Image.new("RGB", (nw + 2 * PAD, TARGET_H + 2 * PAD),
                                 "white")
            canvas_p.paste(im, (PAD, PAD))
            panels.append(canvas_p)
        total_w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
        total_h = max(p.height for p in panels)
        canvas = Image.new("RGB", (total_w, total_h), "white")
        x = 0
        font = get_font(52)
        draw = ImageDraw.Draw(canvas)
        for lab, p in zip(labels, panels):
            canvas.paste(p, (x, 0))
            draw.text((x + 14, 10), lab, fill=(0, 0, 0), font=font)
            x += p.width + GAP
        out = os.path.join(FIG, out_name)
        canvas.save(out)
        print(f"[combo] {out_name}: {canvas.size[0]}x{canvas.size[1]}")
    print("[combo] all done")


if __name__ == "__main__":
    main()
