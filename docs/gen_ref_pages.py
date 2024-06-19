"""Generate the code reference pages and navigation."""

import logging
from pathlib import Path

import griffe
import mkdocs_gen_files

logger = logging.getLogger("gen_ref_pages")
logging.basicConfig(level=logging.INFO)
nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

log_msg = ["\n\tAPI tReference:"]
for path in sorted(
    set(src.rglob("*.py"))
    - set(src.rglob("*tests/*.py"))
    # - set(src.rglob("*/_[!_]*.py"))  # No single underscore file
    - set(src.rglob("*/_[!_]*/*.py"))  # No single underscore module
    - set(src.rglob("__about__.py"))
):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "_config" and parts[-2] == "model_diagnostics":
        # Place _config.py under top level model_diagnostics.
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        g = griffe.load(".".join(parts))
        with mkdocs_gen_files.open(full_doc_path, "a") as fd:
            for f in g.functions:
                ident = ".".join(parts[:-1] + (f,))
                log_msg += [f"Add to {full_doc_path}", f"::: {ident}"]
                fd.write(f"\n::: {ident}")
        continue
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        log_msg += [f"Add to {full_doc_path}", f"::: {ident}"]
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

logger.info("\n\t".join(log_msg))

log_msg = ["\n\tNavigation:"]
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    log_msg += list(nav.build_literate_nav())
    nav_file.writelines(nav.build_literate_nav())

logger.info("\n\t".join(log_msg))
