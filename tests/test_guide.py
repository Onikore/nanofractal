"""Run every Python example of the user guide (guide/*.md).

Blocks of one page run in order in a shared namespace, like a reader pasting
them into one session. A block whose first line is ``# requires: ...`` (a
camera, cv2, ...) is skipped.
"""

import pathlib
import re

import pytest

GUIDE = pathlib.Path(__file__).resolve().parent.parent / "guide"
PAGES = sorted(GUIDE.glob("*.md")) if GUIDE.is_dir() else []
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.skipif(not PAGES, reason="guide/ not available (installed wheel tests)")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_guide_examples_run(page, capsys):
    import nanofractal as nf

    ns: dict = {}
    try:
        for i, code in enumerate(BLOCK.findall(page.read_text(encoding="utf-8"))):
            if code.lstrip().startswith("# requires:"):
                continue
            exec(compile(code, f"{page.name}[block {i}]", "exec"), ns)
    finally:
        nf.set_num_threads(0)  # an example may change the process-wide setting
