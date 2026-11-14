from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import pybind11

_CORE_DIR = pathlib.Path(__file__).resolve().parent
_BINDING_SOURCE = _CORE_DIR / "bindings.cpp"


def _source_files() -> list[pathlib.Path]:
    return sorted(set(_CORE_DIR.glob("*.cpp")) | {_BINDING_SOURCE})


def _digest() -> str:
    hasher = hashlib.sha256()
    hasher.update(sys.version.encode("utf-8"))
    hasher.update(str(sysconfig.get_config_var("EXT_SUFFIX")).encode("utf-8"))
    hasher.update(pybind11.__version__.encode("utf-8"))
    hasher.update(os.environ.get("CXX", "").encode("utf-8"))
    for path in _source_files():
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _compiler() -> str:
    for candidate in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if not candidate:
            continue
        found = shutil.which(candidate)
        if found is not None:
            return found
    raise RuntimeError(
        "no C++ compiler found: install g++/clang++ or set the CXX environment variable"
    )


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ImportError(
            "failed to build the C++ extension for qt_miner:\n"
            + " ".join(command)
            + "\n"
            + result.stderr
        )


def _build() -> pathlib.Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    build_dir = pathlib.Path(tempfile.gettempdir()) / "qt_clustering" / _digest()
    target = build_dir / f"_qt_miner{suffix}"
    if target.exists():
        return target

    build_dir.mkdir(parents=True, exist_ok=True)

    compile_command = [
        _compiler(),
        "-O2",
        "-std=c++17",
        "-I",
        sysconfig.get_paths()["include"],
        "-I",
        pybind11.get_include(),
        "-I",
        str(_CORE_DIR),
    ]
    if sys.platform != "win32":
        compile_command.append("-fPIC")

    object_file = build_dir / "bindings.o"
    compile_command += ["-c", str(_BINDING_SOURCE), "-o", str(object_file)]
    _run(compile_command)

    link_command = [_compiler(), "-shared"]
    if sys.platform == "win32":
        link_command.append("-static")
        libdir = sysconfig.get_config_var("LIBDIR")
        if libdir:
            link_command += ["-L", libdir]
    elif sys.platform == "darwin":
        link_command += ["-undefined", "dynamic_lookup"]

    link_command.append(str(object_file))
    if sys.platform == "win32":
        link_command.append(f"-lpython{sysconfig.get_config_var('VERSION')}")
    link_command += ["-o", str(target)]
    _run(link_command)

    return target


def _load():
    path = _build()
    spec = importlib.util.spec_from_file_location("_qt_miner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_qt_miner = _load()

QTMiner = _qt_miner.QTMiner

QTMiner.__module__ = __name__

__all__ = ["QTMiner"]
