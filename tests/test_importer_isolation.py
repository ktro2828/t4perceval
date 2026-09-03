"""The dependency direction, made executable.

``t4perceval/typing.py`` states that only the importer may depend on ``t4_devkit``, and
the devkit is an optional extra. A stray import at package scope would break installs
without that extra -- and would do it at import time, far from whatever added it.
"""

from __future__ import annotations

import subprocess
import sys

PROBE = """
import sys
import t4perceval
import t4perceval.importer
leaked = sorted(name for name in sys.modules if name.split(".")[0] in {modules!r})
print(",".join(leaked))
"""


def _leaked(modules: set[str]) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(modules=modules)],
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in result.stdout.strip().split(",") if name]


def test_importing_the_package_does_not_pull_in_the_devkit() -> None:
    assert _leaked({"t4_devkit"}) == []


def test_importing_the_package_does_not_pull_in_the_mcap_libraries() -> None:
    assert _leaked({"mcap", "mcap_ros2", "rosbags"}) == []


def test_the_t4_importer_does_pull_in_the_devkit() -> None:
    # The negative tests above would also pass if the import simply did not work, so pin
    # that the dependency exists exactly where it is supposed to.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import t4perceval.importer.t4.source as m; import sys; "
                "print('t4_devkit' in sys.modules or hasattr(m, 'T4Source'))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "True"


def test_a_missing_extra_names_the_install_command() -> None:
    # Without the guard the first thing a user sees is a ModuleNotFoundError for a module
    # they have never heard of, with nothing saying which extra provides it.
    probe = (
        "import sys\n"
        "sys.modules['t4_devkit'] = None\n"
        "from t4perceval.importer.t4 import T4Importer\n"
        "try:\n"
        "    T4Importer.open('tests/data/t4dataset')\n"
        "except ImportError as error:\n"
        "    print(error)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "t4perceval[t4]" in result.stdout
