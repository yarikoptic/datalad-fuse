"""Abstract base for remote file access backends and shared constants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Any, Optional

from .utils import AnnexKey

if TYPE_CHECKING:
    from .adapter import DatasetAdapter

DEFAULT_BACKENDS = "remfile,fsspec"


class Backend(ABC):
    """Base class for remote file access backends."""

    name: str

    @abstractmethod
    def can_handle(self, key: Optional[AnnexKey], mode: str) -> bool:
        """Return True if this backend should be used for *key* in *mode*."""

    def open(
        self,
        adapter: "DatasetAdapter",
        relpath: str,
        key: Optional[AnnexKey],
        mode: str = "rb",
        **kwargs: Any,
    ) -> IO:
        """Open the annexed file *relpath* via this backend.

        The default implementation iterates URLs from
        ``adapter.get_urls(str(key))`` and tries :meth:`open_url` on each in
        order, propagating the last underlying exception if none succeed.
        Backends that don't need URLs at all (e.g. one that fetches the file
        via ``git annex get`` first) should override this method directly.
        """
        last_error: Optional[Exception] = None
        any_url = False
        for url in adapter.get_urls(str(key)):
            any_url = True
            try:
                return self.open_url(url, mode, **kwargs)
            except Exception as e:
                last_error = e
        if last_error is None:
            # `any_url` is False here — no URLs were yielded at all.
            assert not any_url
            raise IOError(f"No URLs available for {relpath} (backend {self.name})")
        # Re-raise the last underlying exception verbatim so the caller's
        # __cause__ chain reflects the actual failure, not a synthetic wrap.
        raise last_error

    def open_url(self, url: str, mode: str = "rb", **kwargs: Any) -> IO:
        """Open *url* and return a file-like object.

        Backends whose :meth:`open` overrides default URL iteration may leave
        this raising ``NotImplementedError`` (the default).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support direct URL access"
        )

    def clear(self) -> None:  # noqa: B027
        """Clear any caches held by this backend.  Default: no-op."""

    def close(self, adapter: "DatasetAdapter") -> None:  # noqa: B027, U100
        """Release backend resources / perform per-session cleanup.

        Called from :meth:`DatasetAdapter.close` for every backend in the
        chain.  Default: no-op.  Backends that have session-scoped state
        (e.g. an annex-get backend that should drop fetched content on exit)
        override this.  *adapter* is passed so callbacks can reach the
        annex / dataset path without holding their own reference.
        """
