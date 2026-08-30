"""Navy/cyan dark theme for the GEIB dashboard.

Single source of styling. :data:`THEME` mirrors the design spec verbatim (the
Plotly Studio token dict) and is the canonical reference for every colour, font,
and spacing token. The named constants below are thin views onto that dict so
the chart/table modules can keep importing short names (``GOLD``, ``NAVY`` …).
Note: ``GOLD`` is retained as the name of the *accent* view for import
compatibility — the accent is now cyan (``#00D9FF``), not gold, and
``DARK_BLUE`` is now the teal button/tag fill (``#0891B2``).
A registered Plotly template (``geib_dark``, composed over ``plotly_dark`` and
set as the default) and a few ``dbc`` layout helpers replace the Dash-Enterprise
``ddk.Card`` shell used by the reference stubs. Page-level chrome (background,
controls, tables, hero) is handled by ``assets/geib.css``, which mirrors the same
tokens as CSS custom properties.
"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from dash import html

# --- Design spec (single source of truth) ----------------------------------
# Verbatim mirror of the Plotly Studio theme export. Treat this dict as the
# authoritative palette/typography reference; the named constants and the Plotly
# template + CSS variables are all derived from these values.
THEME: dict[str, object] = {
    "accent": "#00D9FF",
    "accent_positive": "#10B981",
    "accent_negative": "#EF4444",
    "background_content": "#0F172A",
    "background_page": "#0A0E27",
    "body_text": "#E2E8F0",
    "border": "#334155",
    "button_text": "#FFFFFF",
    "button_background_color": "#0891B2",
    "control_border_color": "#334155",
    "control_background_color": "#1E293B",
    "control_text": "#E2E8F0",
    "card_background_color": "#1E293B",
    "card_box_shadow": "0px 4px 12px rgba(0,0,0,0.3)",
    "card_outline_color": "#334155",
    "card_header_accent": "#0891B2",
    "card_header_background_color": "#1E293B",
    "card_title_text": "#FFFFFF",
    "card_title_font_size": "18px",
    "card_description_text": "#CBD5E1",
    "card_description_font_size": "14px",
    # Categorical slots, in FIXED assignment order -- never cycled, never
    # generated past slot 8. Validated against the card surface (#1E293B) for
    # the OKLCH dark lightness band, a chroma floor, adjacent-pair separation
    # under protanopia/deuteranopia/tritanopia, and contrast.
    #
    # The previous list FAILED that check and it was visible rather than
    # theoretical: slots 1-3 were "#0369A1", "#0891B2", "#06B6D4" -- three steps
    # of one cyan hue. Adjacent normal-vision separation was dE 10.8 against a
    # floor of 15, so a full-colour reader could not reliably tell series 2 from
    # series 3, before any question of colour blindness. Three of the nine also
    # sat outside the dark lightness band.
    #
    # Slot 6 ("#008300") carries a contrast WARN at 2.96:1 against the card. That
    # is legal only because every chart using it also ships a legend or a data
    # table, which is the required relief.
    "colorway": [
        "#3987e5",  # 1 blue
        "#d95926",  # 2 orange
        "#199e70",  # 3 aqua
        "#c98500",  # 4 yellow
        "#d55181",  # 5 magenta
        "#008300",  # 6 green
        "#9085e9",  # 7 violet
        "#e66767",  # 8 red
    ],
    # SEQUENTIAL: one hue, monotonic in lightness. For magnitude -- a quantity
    # with a floor and no meaningful midpoint. Runs mid -> light because the
    # board is dark: the darkest steps of a true light->dark ramp disappear into
    # the #1E293B card.
    "colorscale": [
        "#2a78d6",
        "#3987e5",
        "#5598e7",
        "#6da7ec",
        "#86b6ef",
        "#b7d3f6",
    ],
    # DIVERGING: two hues either side of a NEUTRAL gray midpoint. For polarity --
    # a quantity signed about a meaningful zero (a revision against consensus, an
    # over/underweight). Blue<->orange rather than red<->green, which is the
    # classic pair that collapses under deuteranopia.
    #
    # What this replaces was neither ramp: red -> pale red -> pale blue -> blue
    # with a GREEN tacked on the end, wired to BOTH `sequential` and
    # `sequentialminus`. Three hues, no neutral midpoint, and a diverging shape
    # used for magnitude -- so a plain quantity read as though it had a polarity,
    # and the hue jump at pale-red/pale-blue landed at an arbitrary value rather
    # than at zero.
    "colorscale_diverging": [
        "#1c5cab",
        "#3987e5",
        "#86b6ef",
        "#64748B",
        "#e8a06b",
        "#d95926",
        "#a8410f",
    ],
    "dbc_primary": "#0891B2",
    "dbc_secondary": "#0891B2",
    "dbc_info": "#06B6D4",
    "dbc_gray": "#475569",
    "dbc_success": "#10B981",
    "dbc_warning": "#F59E0B",
    "dbc_danger": "#EF4444",
    "font_family": "Roboto, Segoe UI, Helvetica Neue, sans-serif",
    "font_size": "14px",
    "font_size_smaller_screen": "13px",
    "font_size_header": "20px",
    "section_title_font_size": "20px",
    "footer_background_color": "#0A0E27",
    "footer_title_text": "#FFFFFF",
    "footer_title_font_size": "16px",
    "header_background_color": "#1E293B",
    "header_text": "#E2E8F0",
    "heading_text": "#0F172A",
    "hero_background_color": "#0A0E27",
    "hero_title_text": "#FFFFFF",
    "hero_title_font_size": "42px",
    "hero_subtitle_text": "#CBD5E1",
    "hero_subtitle_font_size": "16px",
    "hero_controls_accent": "#00D9FF",
    "text": "#1E293B",
    "subtle_text": "#CBD5E1",
    "graph_grid_color": "#334155",
    "table_striped_even": "#0F172A",
    "table_striped_odd": "#1E293B",
    "table_border": "#334155",
    "tag_background_color": "#0891B2",
    "tag_border_color": "#06B6D4",
    "tooltip_background_color": "#0A0E27",
    "tooltip_text": "#FFFFFF",
    "tooltip_font_size": "13px",
    "button_border_radius": "6px",
    "button_text_transform": "none",
    "tag_text_color": "#FFFFFF",
    "tag_font_size": "12px",
    "tag_border_radius": "4px",
    "card_menu_background_color": "#1E293B",
    "card_menu_text": "#E2E8F0",
    "card_margin": "20px",
    "card_padding": "8px",
    "hero_controls_background_color": "rgba(10,14,39,0.95)",
    "hero_controls_label_text": "#E2E8F0",
    "hero_controls_label_font_size": "13px",
    "hero_controls_grid_columns": 4,
    "section_padding": "24px",
    "section_gap": "24px",
    "breakpoint_px": "700px",
    "dbc_font_size": "12px",
    "dbc_border": "#334155",
    "color_scheme": "dark",
}

# --- Brand palette (named views onto THEME) --------------------------------
# Name kept as GOLD for import compatibility; the accent is now cyan.
GOLD = THEME["accent"]  # "#00D9FF" — cyan accent
GREEN = THEME["accent_positive"]  # "#10B981"
RED = THEME["accent_negative"]  # "#EF4444"
NAVY = THEME["background_page"]  # "#0A0E27" — page background
BACKGROUND_CONTENT = THEME["background_content"]  # "#0F172A" — plot area / table-even
# Name kept as DARK_BLUE for import compatibility; the fill is now teal.
DARK_BLUE = THEME["button_background_color"]  # "#0891B2" — teal buttons / tags / dbc primary
CONTROL_BG = THEME["control_background_color"]  # "#1E293B" — input / card fields
# Back-compat alias: SLATE_DARK historically named the control/input fill.
SLATE_DARK = CONTROL_BG  # "#1E293B"
BODY_TEXT = THEME["body_text"]  # "#E2E8F0"
SUBTLE_TEXT = THEME["subtle_text"]  # "#CBD5E1"
BORDER = THEME["border"]  # "#334155"
GRID = THEME["graph_grid_color"]  # "#334155" — chart gridlines (spec)
CONTROL_BORDER = THEME["control_border_color"]  # "#334155"
WHITE = THEME["button_text"]  # "#FFFFFF"

FONT_FAMILY = THEME["font_family"]

# Categorical colourway (spec "Chart Colors - Colorway").
COLORWAY = list(THEME["colorway"])

def _plotly_scale(colors: list[str]) -> list[list]:
    """Return *colors* as an evenly-spaced Plotly colorscale."""
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


# One hue, monotonic lightness -- for MAGNITUDE (a floor, no meaningful middle).
_SCALE_COLORS = list(THEME["colorscale"])
SEQUENTIAL_SCALE = _plotly_scale(_SCALE_COLORS)

# Two hues about a neutral gray -- for POLARITY (signed about a real zero).
# Only correct when the colour axis is centred on that zero: pass
# ``cmid=0`` (or symmetric ``cmin``/``cmax``), otherwise the neutral lands at an
# arbitrary value and the ramp claims a polarity the data does not have.
DIVERGING_SCALE = _plotly_scale(list(THEME["colorscale_diverging"]))

#: The two POLES of that ramp, for a binary signed split -- a diverging bar whose
#: bars sit either side of zero, an over/underweight flag. Same hues as the ramp,
#: so a continuous and a binary view of the same quantity agree.
#:
#: Ordered low -> high, matching the ramp: cool for the negative side, warm for
#: the positive. Which end reads as "good" is not encoded here and should not be
#: -- on a diverging bar the SIDE OF THE ZERO LINE already says the sign, and
#: colour is the redundant encoding that makes it survive a colour-blind reader.
DIVERGING_POLES: dict[str, str] = {
    "negative": "#3987e5",
    "positive": "#d95926",
}

#: Deprecated alias kept so existing imports resolve. It was the *diverging*
#: shape wired to Plotly's ``sequential`` slot; it now resolves to the sequential
#: ramp, which is what almost every caller actually wanted. Use
#: :data:`SEQUENTIAL_SCALE` or :data:`DIVERGING_SCALE` explicitly in new code.
CONTINUOUS_SCALE = SEQUENTIAL_SCALE

# --- Named series & translucency -------------------------------------------

def translucent(color: str, alpha: float) -> str:
    """Return *color* as an ``rgba(...)`` string at *alpha*.

    Band fills were being written as a SECOND literal beside the line colour they
    belong to -- ``_FORECAST_COLOR = "rgb(0,100,200)"`` next to
    ``_BAND_FILL = "rgba(0,100,200,0.2)"``. Two literals for one decision drift
    apart the moment either is edited, and a fill that no longer matches its line
    reads as a third series. Derive the fill instead.

    Accepts ``#rgb`` / ``#rrggbb`` / ``rgb(r,g,b)``.
    """
    c = color.strip()
    if c.startswith("rgb("):
        parts = [p.strip() for p in c[4:-1].split(",")]
        r, g, b = (int(float(v)) for v in parts[:3])
    else:
        h = c.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


#: Entities that appear on MORE THAN ONE card, so their hue is fixed centrally.
#:
#: ``pt_convergence`` and ``kalman_structural_forecast`` both plot the realised
#: price against the analyst consensus target, and each had defined its own
#: ``_PRICE_COLOR`` / ``_TARGET_COLOR`` -- the same two literals, written twice.
#: That is the drift hazard the reference-geometry roles were introduced for, in
#: a second place: edit one card and the same entity wears two colours across the
#: board, with nothing to catch it.
#:
#: Drawn from the validated colourway, so these are the same eight hues the
#: categorical charts use rather than a private palette.
SERIES_COLORS: dict[str, str] = {
    "forecast": COLORWAY[0],  # blue   - the model's own output
    "target": COLORWAY[1],    # orange - analyst consensus, the comparison
    "price": COLORWAY[3],     # yellow - realised price (was #FACC15; nearest slot)
}

#: Posterior draws are a texture, not a series: many faint paths whose job is to
#: show spread without competing with the mean drawn over them.
DRAW_COLOR = translucent(SUBTLE_TEXT, 0.18)


# --- Reference geometry ----------------------------------------------------
# SSOT for zero lines, thresholds and target markers, mirroring the
# ``_REF_LINE_KINDS`` / ``_add_ref_line`` convention the Kalman notebook side
# already follows (CLAUDE.md: "never call add_hline / add_vline / add_vrect
# directly").
#
# WHY A ROLE AND NOT A COLOUR. Reference geometry is CONTEXT -- it is closer to
# the gridlines than to the data, and it should recede. Each chart had been
# hand-picking a hue instead, which produced two problems at once:
#
#   Collision. The Piotroski strong/moderate/weak bands were drawn in the status
#   green/amber/red. Those are legitimate status hues, but the categorical
#   colourway also contains a green and a red, so with eight company lines on the
#   chart a reader saw a green dashed rule and a green series and had to work out
#   that only one of them was data.
#
#   Drift. Four different hand-picked colours across the percentile markers in
#   monte_carlo_forecast, three of them hardcoded hexes outside the theme -- and
#   one, "#06B6D4", was slot 3 of the OLD colourway, orphaned when the palette was
#   revalidated. Nothing pointed at it, so nothing updated it.
#
# So reference lines carry no hue of their own. They are separated by DASH
# PATTERN and by their direct label, which is what "Strong (7+)" and "5th %ile"
# were always doing anyway. The one exception is ``emphasis``.
REF_LINE_KINDS: dict[str, dict] = {
    # A baseline the data is signed about -- zero, no-change, break-even. The most
    # recessive: it should be findable and never compete.
    "zero": dict(line_color=BORDER, line_dash="dash", line_width=1, layer="below"),
    # A fixed threshold the reader reads values against: score bands, percentiles,
    # horizon markers. Recessive, but legible against the plot area.
    "anchor": dict(line_color=SUBTLE_TEXT, line_dash="dash", line_width=1, layer="below"),
    # The ONE line a reader is looking for -- their own target, the central
    # estimate. Allowed to read, drawn in the accent, which is deliberately not a
    # categorical slot so it cannot be mistaken for a series.
    "emphasis": dict(line_color=GOLD, line_dash="solid", line_width=2, layer="above"),
}


def ref_line(kind: str = "anchor", **overrides) -> dict:
    """Return ``add_hline`` / ``add_vline`` kwargs for a reference-geometry *kind*.

    Parameters
    ----------
    kind
        One of :data:`REF_LINE_KINDS` -- ``zero``, ``anchor`` or ``emphasis``.
    **overrides
        Merged over the role's defaults; use for ``annotation_text`` and
        ``annotation_position``, not to re-colour the line.

    Raises
    ------
    KeyError
        On an unknown *kind*, rather than silently falling back -- a reference
        line with no role is how the hand-picked colours accumulated.
    """
    if kind not in REF_LINE_KINDS:
        raise KeyError(f"Unknown reference-line kind {kind!r}. "
                       f"Valid: {sorted(REF_LINE_KINDS)}")
    style = dict(REF_LINE_KINDS[kind])
    style.setdefault("annotation_font", dict(size=10, color=SUBTLE_TEXT))
    style.update(overrides)
    return style


# --- Plotly template -------------------------------------------------------
_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=NAVY,
    plot_bgcolor=BACKGROUND_CONTENT,
    font=dict(family=FONT_FAMILY, color=BODY_TEXT, size=13),
    title=dict(font=dict(family=FONT_FAMILY, color=WHITE, size=18)),
    colorway=COLORWAY,
    xaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=BODY_TEXT)),
    colorscale=dict(
        sequential=SEQUENTIAL_SCALE,
        sequentialminus=SEQUENTIAL_SCALE,
        diverging=DIVERGING_SCALE,
    ),
    hoverlabel=dict(bgcolor=NAVY, font=dict(color=WHITE, size=13)),
    margin=dict(l=60, r=30, t=50, b=70),
)
pio.templates["geib_dark"] = _template
pio.templates.default = "plotly_dark+geib_dark"

# --- Layout helpers --------------------------------------------------------
GRAPH_STYLE = {"minHeight": "550px", "height": "calc(100vh - 600px)"}
DUAL_GRAPH_STYLE = {"minHeight": "500px", "height": "calc(100vh - 700px)"}
# Stacked (vertical) variant: each pane spans the full card width, so give the
# figures a fixed height rather than a viewport-relative one (two panes share
# the column, viewport-relative heights would push the second graph off-screen).
STACKED_GRAPH_STYLE = {"minHeight": "420px", "height": "480px"}

CONTROLS_ROW_STYLE = {
    "display": "flex",
    "flexDirection": "row",
    "flexWrap": "wrap",
    "rowGap": "10px",
    "alignItems": "flex-end",
    "marginBottom": "15px",
}


def control(label: str, component: html.Div) -> html.Div:
    """Wrap a labelled control in the standard stacked column layout."""
    return html.Div(
        children=[
            html.Label(label, className="geib-control-label"),
            component,
        ],
        className="geib-control",
    )


def card(
    title: str,
    description: str,
    children: list,
    *,
    card_id: Optional[str] = None,
) -> dbc.Card:
    """Build a themed ``dbc.Card`` matching the ddk Card/Header/Footer shell.

    Parameters
    ----------
    title
        Card title shown in the gold-underlined header.
    description
        Footer description text.
    children
        Card body children (controls, graphs, tables, error blocks).
    card_id
        Optional DOM id for the card.
    """
    kwargs = {"id": card_id} if card_id else {}
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(title, className="geib-card-title"),
                className="geib-card-header",
            ),
            dbc.CardBody(children),
            dbc.CardFooter(description, className="geib-card-desc"),
        ],
        className="geib-card",
        **kwargs,
    )