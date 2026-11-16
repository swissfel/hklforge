"""How a geometry is built, step by step, in the 3D view (a notebook widget).

``GeometryExplorer().widget`` shows a stored geometry (or one passed in,
or YAML typed in) circle by circle: first the lab frame and the beam, then
each sample circle from the outermost in, then the detector circles. At
each step the new circle is highlighted with its axis, and a short text
says what it turns about, what carries it and what it carries. The page's
own sliders move the circles that are there so far.

Needs ipywidgets (imported when the explorer is made, so ``hklforge.view``
itself needs numpy only). Follows the rules that keep a panel rendering
under voila: core widgets only, fixed pixel sizes.
"""
from __future__ import annotations

import html

import numpy as np
import yaml

from ..geometry import Geometry
from . import iframe, message, page, payload_html


def _axis_yaml(v) -> str:
    v = np.asarray(v, float)
    i = int(np.argmax(np.abs(v)))
    if np.isclose(abs(v[i]), 1.0, atol=1e-12):
        return "xyz"[i] + ("+" if v[i] > 0 else "-")
    return "[" + ", ".join(f"{x:.12g}" for x in v) + "]"


def geometry_yaml(g: Geometry) -> str:
    """The geometry as YAML in the style of the stored files (``Geometry.from_yaml`` reads it)."""
    lines = [f"name: {g.name}"]
    if g.description:
        lines.append("description: |")
        lines += ["  " + ln for ln in g.description.rstrip().splitlines()]
    if g.labels:
        lines.append("labels: {" + ", ".join(f"{k}: {v}" for k, v in g.labels.items()) + "}")
    lines.append(f"beam: {_axis_yaml(g.beam)}")
    lines.append(f"up: {_axis_yaml(g.up_direction())}")
    if g.azimuth_zero is not None:
        lines.append(f"azimuth_zero: {_axis_yaml(g.azimuth_zero)}")
    if g.bisect_circle:
        lines.append(f"bisect_circle: {g.bisect_circle}")
    for stack, circles in (("sample", g.sample), ("detector", g.detector)):
        lines.append(f"{stack}:" + ("            # outermost first" if stack == "sample" else ""))
        lines += [f"  - {{name: {c.name}, axis: {_axis_yaml(c.axis)}}}" for c in circles]
    text = "\n".join(lines) + "\n"
    yaml.safe_load(text)               # stays valid YAML
    return text


def _axis_words(g: Geometry, v) -> str:
    v = np.asarray(v, float)
    i = int(np.argmax(np.abs(v)))
    if np.isclose(abs(v[i]), 1.0, atol=1e-9):
        letter = "xyz"[i]
        label = g.labels.get(letter)
        return f"{letter}{'+' if v[i] > 0 else '−'}" + (f" ({label})" if label else "")
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def steps(g: Geometry) -> list[dict]:
    """The build steps: ``{"sample": n, "detector": m, "focus", "title", "text"}``."""
    out = [{"sample": 0, "detector": 0, "focus": None, "title": "the lab frame and the beam",
            "text": (f"The lab frame of <b>{html.escape(g.name)}</b>: the beam runs along "
                     f"<b>{_axis_words(g, g.beam)}</b>, up is <b>{_axis_words(g, g.up_direction())}</b>. "
                     "Every circle is given by its axis in this frame <i>with all angles at 0</i>; "
                     "a positive angle turns right-handed about it (thumb along the axis). "
                     "The circles come in two stacks, each listed outermost first.")}]
    s, d = g.sample, g.detector
    for i, c in enumerate(s):
        inside = [x.name for x in s[i + 1:]]
        text = (f"<b>{c.name}</b> turns about <b>{_axis_words(g, c.axis)}</b> (at all angles 0). ")
        if i == 0:
            text += "It is the outermost sample circle: nothing moves its axis. "
        else:
            text += (f"It sits on <b>{s[i - 1].name}</b>: turning {s[i - 1].name} carries {c.name}'s axis "
                     "with it - move it with the slider and watch the dashed axis. ")
        text += ("It carries " + (", ".join(inside) + " and " if inside else "") + "the sample. "
                 "The sample rotation is the product R = " + " · ".join(f"R({x.name})" for x in s[:i + 1])
                 + (" · ..." if inside else "") + ", outermost on the left.")
        if g.bisect_circle == c.name:
            text += (f" <br>{c.name} is this geometry's <i>bisect circle</i>: the virtual angle ω and the "
                     "bisect condition refer to its axis.")
        out.append({"sample": i + 1, "detector": 0, "focus": c.name,
                    "title": f"sample circle {i + 1} of {len(s)}: {c.name}", "text": text})
    for j, c in enumerate(d):
        text = (f"<b>{c.name}</b> turns about <b>{_axis_words(g, c.axis)}</b> (at all angles 0). "
                "The detector circles turn the outgoing beam: k′ = D k with D = "
                + " · ".join(f"R({x.name})" for x in d[:j + 1]) + (" · ..." if j + 1 < len(d) else "")
                + ". With all detector angles at 0 the detector looks straight along the beam. ")
        if j > 0:
            text += f"It sits on <b>{d[j - 1].name}</b>, which carries its axis. "
        out.append({"sample": len(s), "detector": j + 1, "focus": c.name,
                    "title": f"detector circle {j + 1} of {len(d)}: {c.name}", "text": text})
    out.append({"sample": len(s), "detector": len(d), "focus": None, "title": "the whole geometry",
                "text": (f"{len(s)} sample circle(s) and {len(d)} detector circle(s): "
                         f"{len(s) + len(d)} angles. A reflection fixes 3 numbers (Q in the sample frame), "
                         f"so a finite set of positions needs {len(s) + len(d) - 3} more condition(s) - "
                         "held circles or virtual angles (the solver's <i>mode</i>).")})
    return out


def _truncated(g: Geometry, n_sample: int, n_detector: int) -> Geometry:
    return Geometry(name=g.name, sample=g.sample[:n_sample], detector=g.detector[:n_detector],
                    beam=g.beam, description=g.description, labels=g.labels, up=g.up_direction(),
                    azimuth_zero=g.azimuth_zero,
                    bisect_circle=g.bisect_circle if g.bisect_circle in [c.name for c in g.sample[:n_sample]] else None)


class GeometryExplorer:
    """The widget is ``.widget``. ``geometries``: extra {label: Geometry}
    listed first (e.g. an instrument's own), ``default``: the label shown
    first (default: the first extra one, else psic)."""

    def __init__(self, geometries: dict | None = None, default: str | None = None,
                 view_id: str = "geo3d", height: int = 600):
        import ipywidgets as w

        self.view_id = view_id
        self.geometries = dict(geometries or {})
        for name in Geometry.builtins():
            self.geometries.setdefault(name, Geometry.builtin(name))
        default = default or (next(iter(geometries)) if geometries else "psic")
        self._n = 0
        self._display_sent = False
        L = w.Layout
        self.pick = w.Dropdown(options=list(self.geometries) + ["(YAML below)"], value=default,
                               description="geometry", layout=L(width="300px"))
        self.step = w.IntSlider(value=0, min=0, max=1, description="build step", continuous_update=False,
                                layout=L(width="380px"))
        self.prev = w.Button(description="◀", tooltip="previous step", layout=L(width="40px"))
        self.next = w.Button(description="▶", tooltip="next step", layout=L(width="40px"))
        self.title = w.HTML()
        self.text = w.HTML(layout=L(width="760px"))
        self.table = w.HTML()
        self.view = w.HTML(iframe(page(self.geometries[default], view_id=view_id,
                                       storage="hklforge-geometry", title=None), height, view_id))
        self.holder = w.HTML()
        self.yaml = w.Textarea(layout=L(width="520px", height="260px"))
        self.use = w.Button(description="show this YAML", tooltip="read the YAML above as a geometry")
        self.yaml_msg = w.HTML()
        self.code = w.HTML()

        self.pick.observe(self._on_pick, "value")
        self.step.observe(lambda _c: self._show(), "value")
        self.prev.on_click(lambda _b: setattr(self.step, "value", max(self.step.min, self.step.value - 1)))
        self.next.on_click(lambda _b: setattr(self.step, "value", min(self.step.max, self.step.value + 1)))
        self.use.on_click(self._on_yaml)

        yaml_box = w.VBox([w.HTML("<b>The geometry as YAML</b> - edit it (names, axes as x+/z- or "
                                  "[x, y, z], outermost first) and show it:"),
                           self.yaml, w.HBox([self.use, self.yaml_msg]), self.code])
        self.widget = w.VBox([
            w.HBox([self.pick, self.step, self.prev, self.next]),
            self.title, self.text, self.view, self.holder,
            w.HBox([w.VBox([w.HTML("<b>The circles</b>"), self.table], layout=L(width="560px")), yaml_box]),
        ])
        self._set_geometry(self.geometries[default])

    # ------------------------------------------------------------------ actions
    def _on_pick(self, change):
        if change["new"] in self.geometries:
            self._set_geometry(self.geometries[change["new"]])

    def _on_yaml(self, _b=None):
        try:
            g = Geometry.from_yaml(self.yaml.value)
        except Exception as exc:                      # shown, not raised
            self.yaml_msg.value = f'<span style="color:#b00">{html.escape(str(exc))}</span>'
            return
        self.yaml_msg.value = '<span style="color:#282">ok</span>'
        self.geometries["(YAML below)"] = g
        if self.pick.value != "(YAML below)":
            self.pick.value = "(YAML below)"          # _on_pick shows it
        else:
            self._set_geometry(g, keep_yaml=True)

    def _set_geometry(self, g: Geometry, keep_yaml: bool = False):
        self.geometry = g
        self.steps = steps(g)
        if not keep_yaml:
            self.yaml.value = geometry_yaml(g)
        self.table.value = self._table(g)
        stored = (f", or <code>Geometry.builtin(\"{html.escape(g.name)}\")</code>"
                  if g.name in Geometry.builtins() else "")
        self.code.value = ("<div style='font-size:12px;color:#555'>In Python: "
                           f"<code>Geometry.from_yaml(text)</code>{stored}.</div>")
        self.step.max = len(self.steps) - 1
        if self.step.value == len(self.steps) - 1:
            self._show()
        else:
            self.step.value = len(self.steps) - 1      # the whole geometry first (fires _show)

    def _show(self):
        st = self.steps[min(self.step.value, len(self.steps) - 1)]
        self.title.value = f"<b>Step {self.step.value}: {html.escape(st['title'])}</b>"
        self.text.value = f"<div style='font-size:13px;line-height:1.45'>{st['text']}</div>"
        part = _truncated(self.geometry, st["sample"], st["detector"])
        self._n += 1
        extra = {"focus": st["focus"]}
        if not self._display_sent:                    # once: sliders and axes on
            extra["display"] = {"showAngleControls": True, "showAxes": True}
            self._display_sent = True
        self.holder.value = payload_html(message(part, n=self._n, label=st["title"], **extra), self.view_id)

    @staticmethod
    def _table(g: Geometry) -> str:
        rows = []
        for stack, cs in (("sample", g.sample), ("detector", g.detector)):
            for i, c in enumerate(cs):
                on = cs[i - 1].name if i else "-"
                carries = ", ".join(x.name for x in cs[i + 1:]) or ("the sample" if stack == "sample" else "the detector")
                rows.append(f"<tr><td>{stack}</td><td><b>{html.escape(c.name)}</b></td>"
                            f"<td>{_axis_words(g, c.axis)}</td><td>{html.escape(on)}</td>"
                            f"<td>{html.escape(carries)}</td></tr>")
        head = "".join(f"<th style='text-align:left;padding:2px 8px'>{h}</th>"
                       for h in ("stack", "circle", "axis at 0", "sits on", "carries"))
        return ("<table style='border-collapse:collapse;font-size:12px'>"
                f"<tr>{head}</tr>" + "".join(rows).replace("<td>", "<td style='padding:2px 8px'>") + "</table>"
                f"<div style='font-size:12px;color:#555;margin-top:4px'>beam {_axis_words(g, g.beam)}, "
                f"up {_axis_words(g, g.up_direction())}"
                + (f", bisect circle {html.escape(g.bisect_circle)}" if g.bisect_circle else "") + "</div>")
