"""ament flake8, wired into colcon test rather than a pre-commit hook.

Skipped rather than failed outside a ROS environment: the rest of this suite
runs on a plain Python install, and a lint check that breaks that property
costs more than it is worth.
"""

import pytest


def _describe(error):
    if isinstance(error, str):
        return error
    return (f"{error.filename}:{error.line_number}:{error.column_number}: "
            f"{error.error_code} {error.error_message}")


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    ament_flake8 = pytest.importorskip(
        "ament_flake8.main", reason="ament_flake8 needs a sourced ROS 2 environment"
    )
    rc, errors = ament_flake8.main_with_errors(argv=[])
    assert rc == 0, (f"{len(errors)} flake8 error(s):\n"
                     + "\n".join(_describe(e) for e in errors))
