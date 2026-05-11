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
"""

from __future__ import annotations

import logging
from typing import IO, TYPE_CHECKING, Any, Optional

from .backends import Backend
from .utils import AnnexKey

if TYPE_CHECKING:
    from .adapter import DatasetAdapter

lgr = logging.getLogger("datalad.fuse.annex_get")


class AnnexGetBackend(Backend):
    """Backend that runs ``git annex get`` and serves the local copy.

    Only handles annexed files (``key is not None``) in binary read mode.
    Anything else is delegated to later backends in the chain via
    :meth:`can_handle` returning ``False``.
    """

    name = "annex-get"

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
        lgr.debug("annex-get: %s ready, opening locally", relpath)
        return open(local, mode, **kwargs)  # type: ignore[return-value]
