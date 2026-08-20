"""Style is enforced, not requested.

The standard ament flake8 check, wired into `colcon test` so it runs in CI
rather than in a pre-commit hook someone forgets to install. Unused imports and
undefined names are the two it catches most often, and both are real bugs
rather than style opinions.

Skipped rather than failed outside a ROS environment: the rest of this suite
runs on a plain Python install with no ROS at all, and a lint check that breaks
that property costs more than it is worth.
"""

import pytest


def _describe(error):
    """ament_flake8 returns error objects on some releases and plain strings on
    others. Formatting the wrong one turns a readable style report into an
    AttributeError that hides every finding underneath it."""
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
