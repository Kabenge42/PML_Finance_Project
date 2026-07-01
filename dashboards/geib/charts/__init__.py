"""GEIB analytic chart modules.

Each module exposes a ``component()`` factory returning a themed ``dbc.Card``
and registers its own Dash ``@callback``. Importing the module is enough to
wire the callback into the global Dash callback registry.
"""
