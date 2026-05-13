"""AnnexGetBackend — fetch the entire annex file via ``git annex get``.

Unlike fsspec/remfile, this backend does *not* stream content over HTTP.  On
first access it materialises the whole file locally by running ``git annex
get <relpath>`` and then returns a regular ``open()`` handle to the now-local
path.  Subsequent accesses go through the
:attr:`DatasetAdapter.get_file_state` shortcut (``HAS_CONTENT``) and never
re-enter the backend chain.

Intended use case: pre-fetching a small, known set of referenced files for a
preview build (e.g. the slides linked from a talk) — see
<https://github.com/con/talks/issues/4>.  Trade-offs vs streaming backends:

- ✓ Subsequent reads are local-disk fast and offline-safe.
- ✓ Works for *any* file regardless of format (no HDF5 / Zarr assumption).
- ✗ First access is bounded by the full download time, not first-byte time.
- ✗ Consumes local disk for every accessed file until ``git annex drop``.

Not added to :data:`backends.DEFAULT_BACKENDS` — enable explicitly via
``--backends=annex-get,fsspec`` or the ``datalad.fusefs.backends`` config.

Drop-on-close semantics
-----------------------

By default the backend leaves fetched content on disk (``drop="none"``), so a
follow-up step (CI publish, preview build, etc.) can read it without
re-downloading.  Set ``drop`` to one of:

- ``"none"`` — keep everything (default).
- ``"obtained"`` — when :meth:`close` is called, drop the keys this backend
  fetched during the session.  Files that were already present when the
  session started are untouched.
- ``"all"`` — drop *all* annexed keys in the dataset, regardless of who
  fetched them.  Use with care.

Orthogonally, ``drop_mode`` selects how the drop is performed (matching
``git annex drop`` flags):

- ``"regular"`` — default, verifies a copy exists elsewhere.
- ``"fast"`` — skip the slow consistency check (``--fast``).
- ``"force"`` — drop even if no other copy is known (``--force``).

Both knobs are configurable per-call (``AnnexGetBackend(drop=..., drop_mode=...)``)
or via the datalad config keys ``datalad.fusefs.annex-get.drop`` and
``datalad.fusefs.annex-get.drop-mode`` (read by :func:`adapter.create_backends`).
"""

from __future__ import annotations

import logging
from typing import IO, TYPE_CHECKING, Any, Optional

from .backends import Backend
from .utils import AnnexKey

if TYPE_CHECKING:
    from .adapter import DatasetAdapter

lgr = logging.getLogger("datalad.fuse.annex_get")

_DROP_CHOICES = frozenset({"none", "obtained", "all"})
_DROP_MODE_CHOICES = frozenset({"regular", "fast", "force"})
_DROP_MODE_FLAGS: dict[str, list[str]] = {
    "regular": [],
    "fast": ["--fast"],
    "force": ["--force"],
}


class AnnexGetBackend(Backend):
    """Backend that runs ``git annex get`` and serves the local copy.

    Only handles annexed files (``key is not None``) in binary read mode.
    Anything else is delegated to later backends in the chain via
    :meth:`can_handle` returning ``False``.
    """

    name = "annex-get"

    def __init__(
        self,
        drop: str = "none",
        drop_mode: str = "regular",
    ) -> None:
        if drop not in _DROP_CHOICES:
            raise ValueError(f"drop={drop!r} not in {sorted(_DROP_CHOICES)}")
        if drop_mode not in _DROP_MODE_CHOICES:
            raise ValueError(
                f"drop_mode={drop_mode!r} not in {sorted(_DROP_MODE_CHOICES)}"
            )
        self._drop = drop
        self._drop_mode = drop_mode
        # Relpaths fetched in this session; populated by :meth:`open`, drained
        # by :meth:`close` when ``drop == "obtained"``.
        self._fetched: set[str] = set()

    def can_handle(self, key: Optional[AnnexKey], mode: str) -> bool:  # noqa: U100
        # Need an annex key to know what to `git annex get`, and the
        # post-fetch open() can only return a binary stream from disk.
        return key is not None and mode == "rb"

    def open(
        self,
        adapter: "DatasetAdapter",
        relpath: str,
        key: Optional[AnnexKey],  # noqa: U100
        mode: str = "rb",
        **kwargs: Any,
    ) -> IO:
        if adapter.annex is None:
            raise IOError(
                f"annex-get backend requires an annex repo, but {adapter.path}"
                " is not under git-annex"
            )
        lgr.debug("annex-get: fetching %s", relpath)
        # AnnexRepo.get() raises CommandError on failure; let it propagate
        # so DatasetAdapter.open() can fall through to the next backend.
        adapter.annex.get(relpath)
        local = adapter.path / relpath
        if not local.exists() or local.is_symlink() and not local.resolve().exists():
            # `git annex get` reported success but the object still isn't
            # readable — treat as a backend failure so the chain falls through.
            raise IOError(
                f"annex-get: {relpath} not materialised after `git annex get`"
            )
        # Record the successful fetch so close(drop="obtained") can find it.
        self._fetched.add(relpath)
        lgr.debug("annex-get: %s ready, opening locally", relpath)
        return open(local, mode, **kwargs)  # type: ignore[return-value]

    def close(self, adapter: "DatasetAdapter") -> None:
        """Apply the configured drop-on-close behaviour."""
        if self._drop == "none":
            return
        if adapter.annex is None:
            # Nothing to drop — the dataset isn't even an annex repo.
            return
        flags = _DROP_MODE_FLAGS[self._drop_mode]
        if self._drop == "obtained":
            if not self._fetched:
                lgr.debug("annex-get: nothing fetched this session, no drop")
                return
            files = sorted(self._fetched)
            lgr.info(
                "annex-get: dropping %d obtained file(s) (mode=%s)",
                len(files),
                self._drop_mode,
            )
            adapter.annex.call_annex(["drop", *flags], files=files)
            # Don't re-drop on a second close()
            self._fetched.clear()
        else:  # "all"
            lgr.info(
                "annex-get: dropping ALL annexed content (mode=%s)",
                self._drop_mode,
            )
            adapter.annex.call_annex(["drop", "--all", *flags])
            self._fetched.clear()
