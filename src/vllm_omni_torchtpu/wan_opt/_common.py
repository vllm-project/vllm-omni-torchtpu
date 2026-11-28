# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stdlib-only post-import hook utility for ``wan_opt`` engine submodules.

``apply_engine()`` hooks run from ``apply_omni_tpu_patches()`` at
``OmniTpuPlatform`` construction, which can happen while vLLM-Omni itself is
still importing (platform resolution). Importing vLLM-Omni modules there risks
circular imports, so engine patches register a callback with
``when_imported(name, callback)``: ``callback(module)`` runs right after
``name`` finishes executing (or immediately if it is already imported), in
whichever process imports it.
"""

import importlib.abc
import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any

_PENDING: dict[str, list[Callable[[ModuleType], None]]] = {}
_PENDING_LOCK = threading.RLock()


class _PostImportLoader(importlib.abc.Loader):
    """Wraps a real loader; runs pending callbacks after ``exec_module``."""

    def __init__(self, inner: Any, name: str) -> None:
        self._inner = inner
        self._name = name

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._inner.exec_module(module)
        with _PENDING_LOCK:
            callbacks = _PENDING.pop(self._name, [])
            if not _PENDING and _FINDER in sys.meta_path:
                sys.meta_path.remove(_FINDER)
        for callback in callbacks:
            callback(module)

    def __getattr__(self, name: str) -> Any:  # e.g. get_resource_reader
        return getattr(self._inner, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that wraps the loader of modules with pending hooks."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        with _PENDING_LOCK:
            if fullname not in _PENDING:
                return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostImportLoader(spec.loader, fullname)
                return spec
        return None


_FINDER = _PostImportFinder()


def when_imported(name: str, callback: Callable[[ModuleType], None]) -> None:
    """Run ``callback(module)`` once ``name`` is imported (now, if already loaded).

    Args:
        name: Fully-qualified module name to watch in ``sys.modules``.
        callback: Callable invoked with the imported module object.
    """
    module = sys.modules.get(name)
    if module is not None:
        callback(module)
        return
    with _PENDING_LOCK:
        _PENDING.setdefault(name, []).append(callback)
        if _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)
