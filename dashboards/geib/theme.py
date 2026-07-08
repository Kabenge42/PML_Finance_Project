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
    "colorway": [
        "#0369A1",  # cyan blue
        "#0891B2",  # teal
        "#06B6D4",  # light cyan
        "#10B981",  # green
        "#F59E0B",  # amber
        "#EF4444",  # red
        "#8B5CF6",  # purple
        "#EC4899",  # pink
        "#6366F1",  # indigo
    ],
    "colorscale": [
        "#EF4444",  # red
        "#F87171",  # light red
        "#FCA5A5",  # lighter red
        "#FECACA",  # very light red
        "#FEE2E2",  # pale red
        "#DBEAFE",  # pale blue
        "#BFDBFE",  # light blue
        "#93C5FD",  # soft blue
        "#60A5FA",  # medium blue
        "#10B981",  # green
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

# Sequential colourscale (spec "Chart Colors - Colorscale"): navy -> gold.
_SCALE_COLORS = list(THEME["colorscale"])
CONTINUOUS_SCALE = [
    [i / (len(_SCALE_COLORS) - 1), color] for i, color in enumerate(_SCALE_COLORS)
]

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
    colorscale=dict(sequential=CONTINUOUS_SCALE, sequentialminus=CONTINUOUS_SCALE),
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