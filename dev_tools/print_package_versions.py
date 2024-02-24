from importlib.metadata import version
from importlib.util import find_spec

if __name__ == "__main__":
    for m in ["numpy", "polars", "scipy", "pandas", "pyarrow"]:
        # ruff: noqa: T201
        if find_spec(m):
            print(f"{m} {version(m)}")
        else:
            print(f"{m} not installed")
