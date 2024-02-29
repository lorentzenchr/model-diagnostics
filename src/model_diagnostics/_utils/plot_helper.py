import sys

import matplotlib as mpl


def get_plotly_color(i):
    # Sometimes, those turn out to be the same as matplotlib default.
    # colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    # Those are the plotly color default color palette in hex.
    import plotly.express as px

    colors = px.colors.qualitative.Plotly
    return colors[i % len(colors)]


def get_xlabel(ax):
    if isinstance(ax, mpl.axes.Axes):
        return ax.get_xlabel()
    else:
        # ax = plotly figure
        return ax.layout.xaxis.title.text


def get_ylabel(ax):
    if isinstance(ax, mpl.axes.Axes):
        return ax.get_ylabel()
    else:
        # ax = plotly figure
        return ax.layout.yaxis.title.text


def get_title(ax):
    if isinstance(ax, mpl.axes.Axes):
        return ax.get_title()
    else:
        # ax = plotly figure
        return ax.layout.title.text


def get_legend_list(ax):
    if isinstance(ax, mpl.axes.Axes):
        return [t.get_text() for t in ax.get_legend().get_texts()]
    else:
        # ax = plotly figure
        return [d.name for d in ax.data if d.showlegend is None or d.showlegend]


def is_plotly_figure(x):
    """Return True if the x is a plotly figure."""
    try:
        plotly = sys.modules["plotly"]
    except KeyError:
        return False
    return isinstance(x, plotly.graph_objects.Figure)
