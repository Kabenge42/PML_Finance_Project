"""Navy/gold dark theme for the GEIB dashboard.

Single source of styling: brand colour constants, a registered Plotly template
(``geib_dark``, set as the default and composed over ``plotly_dark``), and a few
``dbc`` layout helpers that replace the Dash-Enterprise ``ddk.Card`` shell used
by the reference stubs. Page-level chrome (background, controls, tables, hero)
is handled by ``assets/geib.css``.
"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from dash import html

# --- Brand palette ---------------------------------------------------------
GOLD = "#D4AF37"
GREEN = "#10B981"
RED = "#EF4444"
NAVY = "#0F172A"
SLATE_DARK = "#1E293B"
BODY_TEXT = "#E2E8F0"
SUBTLE_TEXT = "#CBD5E1"
BORDER = "#334155"
CONTROL_BORDER = "#475569"
WHITE = "#FFFFFF"

FONT_FAMILY = "Segoe UI, Roboto, Helvetica Neue, sans-serif"

# Categorical colourway (spec "Chart Colors - Colorway").
COLORWAY = [
    "#1E3A8A",  # dark blue
    "#D4AF37",  # gold
    "#10B981",  # green
    "#0EA5E9",  # sky blue
    "#8B5CF6",  # purple
    "#EC4899",  # pink
    "#F59E0B",  # amber
    "#06B6D4",  # cyan
    "#6366F1",  # indigo
]

# Sequential colourscale (spec "Chart Colors - Colorscale"): navy -> gold.
_SCALE_COLORS = [
    "#0F172A",
    "#1E3A8A",
    "#1E40AF",
    "#2563EB",
    "#3B82F6",
    "#60A5FA",
    "#93C5FD",
    "#DBEAFE",
    "#F0F9FF",
    "#D4AF37",
]
CONTINUOUS_SCALE = [
    [i / (len(_SCALE_COLORS) - 1), color] for i, color in enumerate(_SCALE_COLORS)
]

# --- Plotly template -------------------------------------------------------
_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=NAVY,
    plot_bgcolor=SLATE_DARK,
    font=dict(family=FONT_FAMILY, color=BODY_TEXT, size=13),
    title=dict(font=dict(family=FONT_FAMILY, color=WHITE, size=18)),
    colorway=COLORWAY,
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
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
