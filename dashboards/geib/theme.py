"""Navy/gold dark theme for the GEIB dashboard.

Single source of styling. :data:`THEME` mirrors the design spec verbatim (the
Plotly Studio token dict) and is the canonical reference for every colour, font,
and spacing token. The named constants below are thin views onto that dict so
the chart/table modules can keep importing short names (``GOLD``, ``NAVY`` …).
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
    "accent": "#D4AF37",
    "accent_positive": "#10B981",
    "accent_negative": "#EF4444",
    "background_content": "#1A2332",
    "background_page": "#0F172A",
    "body_text": "#E2E8F0",
    "border": "#334155",
    "button_text": "#FFFFFF",
    "button_background_color": "#1E3A8A",
    "control_border_color": "#475569",
    "control_background_color": "#1E293B",
    "control_text": "#E2E8F0",
    "card_background_color": "#F8FAFC",
    "card_box_shadow": "0px 4px 12px rgba(0,0,0,0.3)",
    "card_outline_color": "#334155",
    "card_header_accent": "#D4AF37",
    "card_header_background_color": "#1A2332",
    "card_title_text": "#FFFFFF",
    "card_title_font_size": "18px",
    "card_description_text": "#CBD5E1",
    "card_description_font_size": "14px",
    "colorway": [
        "#1E3A8A",  # dark blue
        "#D4AF37",  # gold
        "#10B981",  # green
        "#0EA5E9",  # sky blue
        "#8B5CF6",  # purple
        "#EC4899",  # pink
        "#F59E0B",  # amber
        "#EF4444",  # red
        "#6366F1",  # indigo
    ],
    "colorscale": [
        "#0F172A",  # navy
        "#1E3A8A",  # dark blue
        "#1E40AF",  # blue
        "#2563EB",  # bright blue
        "#3B82F6",  # medium blue
        "#60A5FA",  # light blue
        "#93C5FD",  # lighter blue
        "#DBEAFE",  # very light blue
        "#F0F9FF",  # almost-white blue
        "#D4AF37",  # gold
    ],
    "dbc_primary": "#1E3A8A",
    "dbc_secondary": "#64748B",
    "dbc_info": "#0EA5E9",
    "dbc_gray": "#94A3B8",
    "dbc_success": "#10B981",
    "dbc_warning": "#F59E0B",
    "dbc_danger": "#EF4444",
    "font_family": "Segoe UI, Roboto, Helvetica Neue, sans-serif",
    "font_size": "14px",
    "font_size_smaller_screen": "13px",
    "font_size_header": "28px",
    "section_title_font_size": "22px",
    "footer_background_color": "#0F172A",
    "footer_title_text": "#FFFFFF",
    "footer_title_font_size": "16px",
    "header_background_color": "#1A2332",
    "header_text": "#FFFFFF",
    "heading_text": "#0F172A",
    "hero_background_color": "#0F172A",
    "hero_title_text": "#FFFFFF",
    "hero_title_font_size": "42px",
    "hero_subtitle_text": "#CBD5E1",
    "hero_subtitle_font_size": "16px",
    "hero_controls_accent": "#D4AF37",
    "text": "#1E293B",
    "subtle_text": "#CBD5E1",
    "graph_grid_color": "#E2E8F0",
    "table_striped_even": "#1A2332",
    "table_striped_odd": "#0F172A",
    "table_border": "#334155",
    "tag_background_color": "#1E3A8A",
    "tag_border_color": "#3B82F6",
    "tooltip_background_color": "#0F172A",
    "tooltip_text": "#FFFFFF",
    "tooltip_font_size": "13px",
    "button_border_radius": "6px",
    "button_text_transform": "none",
    "tag_text_color": "#1E3A8A",
    "tag_font_size": "12px",
    "tag_border_radius": "4px",
    "card_menu_background_color": "#1A2332",
    "card_menu_text": "#E2E8F0",
    "card_margin": "20px",
    "card_padding": "16px",
    "hero_controls_background_color": "rgba(15,23,42,0.95)",
    "hero_controls_label_text": "#E2E8F0",
    "hero_controls_label_font_size": "13px",
    "hero_controls_grid_columns": 4,
    "section_padding": "24px",
    "section_gap": "24px",
    "breakpoint_px": "700px",
    "dbc_font_size": "12px",
    "dbc_border": "#E2E8F0",
    "color_scheme": "dark",
}

# --- Brand palette (named views onto THEME) --------------------------------
GOLD = THEME["accent"]  # "#D4AF37"
GREEN = THEME["accent_positive"]  # "#10B981"
RED = THEME["accent_negative"]  # "#EF4444"
NAVY = THEME["background_page"]  # "#0F172A" — page background
BACKGROUND_CONTENT = THEME["background_content"]  # "#1A2332" — cards / headers / table-even
DARK_BLUE = THEME["button_background_color"]  # "#1E3A8A" — buttons / tags / dbc primary
CONTROL_BG = THEME["control_background_color"]  # "#1E293B" — input fields
# Back-compat alias: SLATE_DARK historically named the control/input fill.
SLATE_DARK = CONTROL_BG  # "#1E293B"
BODY_TEXT = THEME["body_text"]  # "#E2E8F0"
SUBTLE_TEXT = THEME["subtle_text"]  # "#CBD5E1"
BORDER = THEME["border"]  # "#334155"
GRID = THEME["graph_grid_color"]  # "#E2E8F0" — chart gridlines (spec)
CONTROL_BORDER = THEME["control_border_color"]  # "#475569"
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