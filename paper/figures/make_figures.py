"""Paper figures as hand-laid SVG (<= 8 shapes, no crossing arrows), rasterized at 10x."""
from pathlib import Path
import cairosvg

OUT = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"
C = {"blue": ("#dbeafe", "#1d4ed8"), "green": ("#dcfce7", "#15803d"), "amber": ("#fef3c7", "#b45309"),
     "red": ("#fee2e2", "#b91c1c"), "indigo": ("#e0e7ff", "#4338ca"), "slate": ("#f1f5f9", "#475569")}


def box(x, y, w, h, color, title, lines=()):
    fill, stroke = C[color]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>']
    n = 1 + len(lines)
    top = y + h / 2 - (n - 1) * 6.5
    out.append(f'<text x="{x + w / 2}" y="{top + 3.5}" text-anchor="middle" font-family="{FONT}" font-size="9.5" font-weight="bold" fill="#0f172a">{title}</text>')
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + w / 2}" y="{top + 3.5 + 13 * (i + 1)}" text-anchor="middle" font-family="{FONT}" font-size="8.5" fill="#334155">{line}</text>')
    return out


def arrow(points, label=None, lx=None, ly=None, color="#334155", dashed=False):
    d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    out = [f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.2" marker-end="url(#arr)"{dash}/>']
    if label:
        w = 5.0 * len(label) + 8
        out.append(f'<rect x="{lx - w / 2}" y="{ly - 7.5}" width="{w}" height="12" rx="3" fill="#ffffff"/>')
        out.append(f'<text x="{lx}" y="{ly + 1.5}" text-anchor="middle" font-family="{FONT}" font-size="8.5" fill="#334155">{label}</text>')
    return out


def svg(name, height, parts):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="{height}" viewBox="0 0 500 {height}">'
            '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#334155"/></marker></defs>'
            f'<rect width="500" height="{height}" fill="#ffffff"/>')
    text = head + "".join(parts) + "</svg>"
    (OUT / f"{name}.svg").write_text(text, encoding="utf-8")
    cairosvg.svg2png(bytestring=text.encode(), write_to=str(OUT / f"{name}.png"), scale=10)


def figure1():
    p = []
    p += box(15, 20, 130, 58, "blue", "Policy runtime", ["frozen; selects an", "admitted skill per task"])
    p += box(185, 20, 130, 58, "slate", "Trajectory log", ["tool calls, bindings,", "outputs, violations"])
    p += box(355, 20, 130, 58, "indigo", "Skill compiler", ["trajectory to program", "with version pins"])
    p += box(355, 128, 130, 62, "green", "Admission gate", ["held-out replay; returns", "accept, reject, or insufficient"])
    p += box(170, 128, 125, 62, "indigo", "Skill graph", ["admitted skills, checked", "edges, revocation"])
    p += box(355, 238, 130, 52, "amber", "Evidence store", ["SHA-256 manifests,", "signature, anchor"])
    p += box(15, 128, 130, 62, "slate", "Independent audit", ["re-executes decisions", "from bundles"])
    p += arrow([(145, 49), (183, 49)])
    p += arrow([(315, 49), (353, 49)])
    p += arrow([(420, 78), (420, 126)], "candidate", 420, 103)
    p += arrow([(355, 159), (297, 159)], "accept", 326, 150)
    p += arrow([(232, 128), (232, 104), (80, 104), (80, 80)], "available skills", 156, 104)
    p += arrow([(420, 190), (420, 236)], "bundle", 420, 214)
    p += arrow([(355, 264), (80, 264), (80, 192)], "bundles and anchors", 220, 264)
    svg("figure1_architecture", 305, p)


def figure2():
    p = []
    p += box(15, 20, 100, 52, "indigo", "Candidate", ["compiled program", "and claimed digest"])
    p += box(140, 20, 100, 52, "slate", "Pre-checks", ["digest, pins,", "test coverage"])
    p += box(265, 20, 100, 52, "green", "Replay tests", ["exact, or Clopper-", "Pearson bound vs tau"])
    p += box(390, 20, 95, 52, "blue", "Admitted", ["callable in", "the skill graph"])
    p += box(140, 112, 225, 44, "red", "Reject or insufficient evidence", ["reason codes are recorded in the bundle"])
    p += box(390, 112, 95, 44, "amber", "Revoked", ["with dependents"])
    p += arrow([(115, 46), (138, 46)])
    p += arrow([(240, 46), (263, 46)])
    p += arrow([(365, 46), (388, 46)])
    p += arrow([(190, 72), (190, 110)], "fail", 190, 92)
    p += arrow([(315, 72), (315, 110)], "fail", 315, 92)
    p += arrow([(437, 72), (437, 110)], "tool change or violation", 437, 92)
    p += arrow([(437, 156), (437, 182), (40, 182), (40, 74)], "re-verify against current tool versions", 240, 182)
    svg("figure2_lifecycle", 196, p)


def figure3():
    p = []
    p += box(15, 22, 150, 50, "indigo", "prep_mL", ["Post: unit = mL,", "validated = true"])
    p += box(330, 22, 155, 50, "blue", "render_liters", ["Pre: unit = L,", "validated = true"])
    p += arrow([(165, 47), (328, 47)], "refuted: unit mL is not L", 246, 38, color="#b91c1c")
    p += box(15, 108, 150, 50, "indigo", "prep_mL", ["Post: unit = mL,", "validated = true"])
    p += box(190, 108, 120, 50, "slate", "rescale_mL_L", ["Pre: unit = mL", "Post: unit = L, inherit"])
    p += box(330, 108, 155, 50, "blue", "render_liters", ["Pre: unit = L,", "validated = true"])
    p += arrow([(165, 133), (188, 133)])
    p += arrow([(310, 133), (328, 133)])
    p += [f'<text x="250" y="182" text-anchor="middle" font-family="{FONT}" font-size="8.5" fill="#15803d">every Pre holds in the propagated state: chain accepted</text>']
    svg("figure3_composition", 196, p)


if __name__ == "__main__":
    figure1(); figure2(); figure3()
    print("ok")
