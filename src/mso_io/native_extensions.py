"""native_extensions.py -- inventory + rebuild helper for this repo's
pybind11-compiled .so extensions (yamlwriter, parallel_yaml_writer, shuffle).

KRUNAL_JULY_23.md point 8 asked for a ".so file generator" GUI. The original
scoping assumption (see the approved plan) was that source might not exist
at all ("the tree currently ships prebuilt .so with no source ... flag as
dependent on that"). It does exist -- found under config_tools/ -- each with
its own build.sh/setup.py and a build/ directory from a prior successful
build. This module is read-only discovery + a rebuild *command* builder;
it deliberately does NOT invoke cmake/make itself from here (framework-free,
no subprocess) -- the GUI layer runs the returned command via the existing
subprocess/thread machinery so failures show up as normal captured output,
same as every other backend call in this codebase.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# One canonical build location per extension (the repo has some duplicate
# copies under config_tools/ -- these are the ones with a build.sh/setup.py
# AND a prior successful build/ directory, i.e. proven to work here).
_BUILD_SOURCES = {
    "yamlwriter": REPO_ROOT / "config_tools" / "converters_io" / "YAML_WRITER",
    "parallel_yaml_writer": REPO_ROOT / "config_tools" / "parallel_native" / "PARALLEL_YAML_WRITER",
    "shuffle": REPO_ROOT / "config_tools" / "data_generators" / "GENERATE_SHUFFLED_VALUES_IN_DESIGNMATRIX",
}

# Runtime locations that need a fresh copy of the .so after a rebuild (every
# directory sens_3_params.py / simulation_manager2_0.py / MUQ-SAC's own
# drivers import these from via sys.path.append(".../X.so"), confirmed by
# grep across src/MUQ-SAC, src/Sensitivity_Analysis, src/MUQ-SAC_EXTENDED).
_RUNTIME_TARGETS = [
    REPO_ROOT / "src" / "MUQ-SAC",
    REPO_ROOT / "src" / "Sensitivity_Analysis",
    REPO_ROOT / "src" / "MUQ-SAC_EXTENDED",
]


@dataclass
class ExtensionInfo:
    name: str
    source_dir: str
    has_source: bool
    has_build_script: bool
    build_command: List[str]
    installed_copies: List[str] = field(default_factory=list)
    built_artifact_glob: str = ""


def _find_installed_copies(name: str) -> List[str]:
    found = []
    for target_dir in _RUNTIME_TARGETS:
        for pattern in (f"{name}.so", f"{name}.cpython-*.so"):
            found += [str(p) for p in target_dir.glob(pattern)]
    return sorted(found)


def inspect_extension(name: str) -> ExtensionInfo:
    if name not in _BUILD_SOURCES:
        raise ValueError(f"Unknown extension {name!r}; known: {sorted(_BUILD_SOURCES)}")
    src_dir = _BUILD_SOURCES[name]
    has_source = any(src_dir.glob("*.cpp"))
    build_sh = src_dir / "build.sh"
    setup_py = src_dir / "setup.py"

    if build_sh.exists():
        build_command = ["bash", str(build_sh)]
        has_build_script = True
    elif setup_py.exists():
        build_command = ["python3", str(setup_py), "build_ext", "--inplace"]
        has_build_script = True
    else:
        build_command = []
        has_build_script = False

    return ExtensionInfo(
        name=name,
        source_dir=str(src_dir),
        has_source=has_source,
        has_build_script=has_build_script,
        build_command=build_command,
        installed_copies=_find_installed_copies(name),
        built_artifact_glob=f"{name}*.so",
    )


def inspect_all() -> List[ExtensionInfo]:
    return [inspect_extension(name) for name in sorted(_BUILD_SOURCES)]


def find_built_artifact(info: ExtensionInfo) -> Optional[str]:
    """After a rebuild, locate the produced .so under the source dir (CMake
    puts it in build/, setuptools puts it next to setup.py or in build/lib*/)."""
    src_dir = Path(info.source_dir)
    candidates = (
        glob.glob(str(src_dir / info.built_artifact_glob))
        + glob.glob(str(src_dir / "build" / info.built_artifact_glob))
        + glob.glob(str(src_dir / "build" / "lib*" / info.built_artifact_glob))
    )
    return candidates[0] if candidates else None


def copy_to_runtime_targets(built_artifact: str, name: str) -> List[str]:
    """Copy a freshly-built .so to every runtime location that needs it.
    Returns the list of destination paths written."""
    import shutil
    written = []
    for target_dir in _RUNTIME_TARGETS:
        if not target_dir.exists():
            continue
        dest = target_dir / f"{name}.so"
        shutil.copy2(built_artifact, str(dest))
        written.append(str(dest))
    return written
