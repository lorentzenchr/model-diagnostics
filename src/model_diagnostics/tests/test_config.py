import pytest

from model_diagnostics import config_context, get_config, set_config


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [
        ("plot_backend", "XXX", "The plot_backend must be"),
    ],
)
def test_set_config_raises(param, value, msg):
    """Test that set_config raises errors."""
    with pytest.raises(ValueError, match=msg):
        set_config(**{param: value})


def test_config_context():
    # Default value.
    assert get_config() == {"plot_backend": "matplotlib"}

    pytest.importorskip("plotly")

    # Not using as a context manager affects nothing
    config_context(plot_backend="plotly")
    assert get_config()["plot_backend"] == "matplotlib"

    with config_context(plot_backend="plotly"):
        assert get_config() == {"plot_backend": "plotly"}
    assert get_config()["plot_backend"] == "matplotlib"

    with config_context(plot_backend="plotly"):
        with config_context(plot_backend=None):
            assert get_config()["plot_backend"] == "plotly"

        assert get_config()["plot_backend"] == "plotly"

        with config_context(plot_backend="matplotlib"):
            assert get_config()["plot_backend"] == "matplotlib"

            with config_context(plot_backend=None):
                assert get_config()["plot_backend"] == "matplotlib"

                # global setting will not be retained outside of context that
                # did not modify this setting
                set_config(plot_backend="plotly")
                assert get_config()["plot_backend"] == "plotly"

            assert get_config()["plot_backend"] == "matplotlib"

        assert get_config()["plot_backend"] == "plotly"

    assert get_config() == {"plot_backend": "matplotlib"}

    # No positional arguments
    with pytest.raises(TypeError):
        config_context(True)  # ruff: noqa: FBT003

    # No unknown arguments
    with pytest.raises(TypeError):
        config_context(do_something_else=True).__enter__()
