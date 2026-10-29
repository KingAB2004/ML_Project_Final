"""Tiny SVG diagram kit: boxes on a grid, orthogonal arrows, panels. Rendered to PNG with headless Chrome."""
import html, subprocess, textwrap
from pathlib import Path

FONT = "'DejaVu Sans', 'Noto Sans', Arial, sans-serif"
INK, MUTED, LINE = "#1d1d1f", "#5b5b63", "#8a8a93"
KIND = {  # fill, stroke, title colour
    "data":    ("#eaf2fd", "#2a78d6", "#14498a"),
    "model":   ("#fdeee6", "#eb6834", "#9a3b12"),
    "agent":   ("#efe9fb", "#8e5bd6", "#4f2a8f"),
    "memory":  ("#e6f6ef", "#1baf7a", "#0d6b49"),
    "eval":    ("#fdf5e1", "#d99a00", "#7a5600"),
    "result":  ("#f1f1f3", "#55555d", "#222226"),
    "gate":    ("#fde9ec", "#d6455d", "#8c1d30"),
    "planned": ("#ffffff", "#9a9aa3", "#55555d"),
    "plain":   ("#ffffff", "#c9c9cf", "#222226"),
}
CW = 0.56  # average character width as a fraction of font size


class Diagram:
    def __init__(self, w, h, title, subtitle=""):
        self.w, self.h, self.parts = w, h, []
        self.text(28, 40, title, 24, INK, bold=True)
        if subtitle:
            self.text(28, 66, subtitle, 13.5, MUTED)

    def text(self, x, y, s, size=13, color=INK, bold=False, anchor="start", italic=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                          f'xml:space="preserve" font-weight="{700 if bold else 400}" font-style="{"italic" if italic else "normal"}">'
                          f'{html.escape(s)}</text>')

    def panel(self, x, y, w, h, label, color="#f6f6f8", stroke="#dcdce1", dashed=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{color}" '
                          f'stroke="{stroke}" stroke-width="1.3"{dash}/>')
        self.text(x + 16, y + 24, label, 14, MUTED, bold=True)

    def box(self, x, y, w, h, title, lines=(), kind="plain", size=12, tag=None, align="left", wrap=True):
        fill, stroke, tcol = KIND[kind]
        dash = ' stroke-dasharray="6 4"' if kind == "planned" else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1.6"{dash}/>')
        cx = x + w / 2
        tx, anchor = (cx, "middle") if align == "center" else (x + 12, "start")
        ty = y + 21
        for t in textwrap.wrap(title, int((w - 24) / (CW * 13.5))) or [title]:
            self.text(tx, ty, t, 13.5, tcol, bold=True, anchor=anchor)
            ty += 17
        ty += 2
        maxc = int((w - 24) / (CW * size))
        for ln in lines:
            bullet = ln.startswith("• ")
            body = ln[2:] if bullet else ln
            wrapped = textwrap.wrap(body, maxc - (2 if bullet else 0)) if wrap else [body]
            for i, part in enumerate(wrapped or [""]):
                prefix = ("• " if i == 0 else "   ") if bullet else ""
                self.text(tx, ty, prefix + part, size, INK, anchor=anchor)
                ty += size * 1.32
        if ty - y > h + 2:
            print(f"  ! overflow in box '{title}': needs {ty - y:.0f}px, has {h}px")
        if tag:
            tw = len(tag) * CW * 10.5 + 14
            self.parts.append(f'<rect x="{x + w - tw - 6}" y="{y - 9}" width="{tw}" height="17" rx="8.5" '
                              f'fill="{stroke}"/>')
            self.text(x + w - tw / 2 - 6, y + 3.5, tag, 10.5, "#ffffff", bold=True, anchor="middle")
        return (x, y, w, h)

    def arrow(self, pts, label=None, dashed=False, color=LINE, lpos=None, lsize=11.5, both=False):
        d = " ".join(f"{'M' if i == 0 else 'L'}{px},{py}" for i, (px, py) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        start = ' marker-start="url(#ah)"' if both else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.7"{dash} '
                          f'marker-end="url(#ah)"{start}/>')
        if label:
            if lpos is None:
                (x1, y1), (x2, y2) = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
                lpos = ((x1 + x2) / 2, (y1 + y2) / 2 - 6)
            tw = len(label) * CW * lsize + 10
            self.parts.append(f'<rect x="{lpos[0] - tw / 2}" y="{lpos[1] - lsize}" width="{tw}" '
                              f'height="{lsize + 6}" rx="4" fill="#ffffff" opacity="0.92"/>')
            self.text(lpos[0], lpos[1], label, lsize, MUTED, anchor="middle", italic=True)

    def legend(self, x, y, items):
        for i, (kind, label) in enumerate(items):
            fill, stroke, _ = KIND[kind]
            dash = ' stroke-dasharray="5 3"' if kind == "planned" else ""
            self.parts.append(f'<rect x="{x}" y="{y + i * 22}" width="22" height="14" rx="4" fill="{fill}" '
                              f'stroke="{stroke}" stroke-width="1.5"{dash}/>')
            self.text(x + 30, y + i * 22 + 11.5, label, 12, MUTED)

    def save(self, out: Path):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
               f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>'
               f'<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.parts) + "</svg>")
        out.with_suffix(".svg").write_text(svg)
        page = out.with_suffix(".html")
        page.write_text(f'<html><body style="margin:0;background:#fff">{svg}</body></html>')
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--window-size={self.w},{self.h}", "--force-device-scale-factor=2",
                        f"--screenshot={out.with_suffix('.png')}", page.as_uri()],
                       check=True, capture_output=True)
        page.unlink()
        print("wrote", out.with_suffix(".png"))
