"""Report files: JSON (source of truth), Markdown tables and SVG charts (SPEC section 12.5)."""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")


def write_json(name: str, data: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / name
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def write_text(name: str, text: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def usd(x: float, digits: int = 6) -> str:
    return f"${x:.{digits}f}"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --- SVG line chart ---------------------------------------------------------------


def svg_sweep(
    title: str,
    xs: list[float],
    series: list[tuple[str, list[float], str, bool]],
    marker_x: float | None = None,
    marker_label: str = "",
    y_label: str = "rate",
) -> str:
    """A small dependency-free line chart. series = (label, ys, colour, dashed)."""
    w, h = 720, 400
    left, right, top, bottom = 60, 180, 40, 50
    pw, ph = w - left - right, h - top - bottom
    x0, x1 = min(xs), max(xs)

    def px(x):
        return left + (x - x0) / (x1 - x0) * pw

    def py(y):
        return top + (1 - y) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="#111">{title}</text>',
    ]
    for i in range(6):  # horizontal grid every 20 %
        y = i / 5
        parts.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{py(y):.1f}" y2="{py(y):.1f}" '
            f'stroke="#e5e7eb"/><text x="{left - 8}" y="{py(y) + 4:.1f}" text-anchor="end" '
            f'fill="#555">{int(y * 100)}%</text>'
        )
    step = 0.05
    t = x0
    while t <= x1 + 1e-9:
        parts.append(
            f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{top + ph}" y2="{top + ph + 5}" '
            f'stroke="#555"/><text x="{px(t):.1f}" y="{top + ph + 20}" text-anchor="middle" '
            f'fill="#555">{t:.2f}</text>'
        )
        t = round(t + step, 2)
    parts.append(
        f'<text x="{left + pw / 2}" y="{h - 8}" text-anchor="middle" fill="#333">'
        "similarity threshold</text>"
    )
    parts.append(
        f'<text x="16" y="{top + ph / 2}" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90 16 {top + ph / 2})">{y_label}</text>'
    )
    if marker_x is not None:
        parts.append(
            f'<line x1="{px(marker_x):.1f}" x2="{px(marker_x):.1f}" y1="{top}" '
            f'y2="{top + ph}" stroke="#111" stroke-dasharray="2 3"/>'
            f'<text x="{px(marker_x) + 4:.1f}" y="{top + 12}" fill="#111">{marker_label}</text>'
        )
    for i, (label, ys, colour, dashed) in enumerate(series):
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys, strict=True))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>'
        )
        ly = top + 10 + i * 22
        parts.append(
            f'<line x1="{left + pw + 16}" x2="{left + pw + 40}" y1="{ly}" y2="{ly}" '
            f'stroke="{colour}" stroke-width="2"{dash}/>'
            f'<text x="{left + pw + 46}" y="{ly + 4}" fill="#222">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
