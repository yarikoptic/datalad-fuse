"""Tests for the AnnexGetBackend (no-network, monkey-patched annex)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional
from unittest.mock import MagicMock

import pytest

from datalad_fuse.adapter import DatasetAdapter, FileState, create_backends
from datalad_fuse.annex_get import AnnexGetBackend
from datalad_fuse.utils import AnnexKey


@pytest.mark.ai_generated
class TestAnnexGetCanHandle:
    """Annex-get only handles annexed files in binary read mode."""

    def setup_method(self) -> None:
        self.backend = AnnexGetBackend()
        self.key = AnnexKey(backend="MD5E", name="abc", size=100, suffix=".md")

    def test_accepts_annexed_binary(self) -> None:
        assert self.backend.can_handle(self.key, "rb") is True

    def test_rejects_non_annexed(self) -> None:
        assert self.backend.can_handle(None, "rb") is False

    @pytest.mark.parametrize("mode", ["r", "rt", "w", "wb", "ab"])
    def test_rejects_non_binary_mode(self, mode: str) -> None:
        assert self.backend.can_handle(self.key, mode) is False

    def test_handles_any_suffix(self) -> None:
        """Unlike remfile, annex-get is format-agnostic."""
        for suffix in (".md", ".pdf", ".png", ".unknown", None):
            key = AnnexKey(backend="MD5E", name="x", size=1, suffix=suffix)
            assert self.backend.can_handle(key, "rb") is True


def _make_dataset_adapter(
    tmp_path: Path,
    backends: list,
    fake_urls: list[str],
    annex: Optional[MagicMock] = None,
) -> DatasetAdapter:
    """Build a DatasetAdapter whose file-state, URLs, and annex are stubbed."""
    adapter = DatasetAdapter.__new__(DatasetAdapter)
    adapter.path = tmp_path
    adapter.mode_transparent = False
    adapter.annex = annex
    adapter._backends = backends

    key = AnnexKey(backend="MD5E", name="abc", size=100, suffix=".md")

    def fake_get_file_state(_relpath: str):
        return (FileState.NO_CONTENT, key)

    def fake_get_urls(_key: str) -> Iterator[str]:
        yield from fake_urls

    adapter.get_file_state = fake_get_file_state  # type: ignore[method-assign]
    adapter.get_urls = fake_get_urls  # type: ignore[method-assign]
    return adapter


@pytest.mark.ai_generated
class TestAnnexGetOpen:
    """Exercise the side effects of AnnexGetBackend.open."""

    def test_calls_annex_get_then_opens_local(self, tmp_path: Path) -> None:
        """annex.get(relpath) is invoked, then the local file is opened."""
        # Pre-populate the "fetched" file so the post-fetch open() succeeds.
        (tmp_path / "doc.md").write_bytes(b"hello annex-get\n")
        annex = MagicMock()
        annex.get = MagicMock(return_value=None)
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)

        with adapter.open("doc.md") as f:
            assert f.read() == b"hello annex-get\n"
        annex.get.assert_called_once_with("doc.md")

    def test_raises_when_dataset_has_no_annex(self, tmp_path: Path) -> None:
        """Plain git repo (annex=None) surfaces a clear IOError via __cause__."""
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=None)

        with pytest.raises(IOError) as exc_info:
            adapter.open("doc.md")
        # DatasetAdapter wraps the backend's IOError; the original is in __cause__
        assert "not under git-annex" in str(exc_info.value.__cause__)

    def test_raises_when_file_missing_after_get(self, tmp_path: Path) -> None:
        """annex.get() returns successfully but file isn't materialised."""
        annex = MagicMock()
        annex.get = MagicMock(return_value=None)
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)
        # No file pre-populated → open() will see local.exists() == False.

        with pytest.raises(IOError) as exc_info:
            adapter.open("doc.md")
        assert "not materialised" in str(exc_info.value.__cause__)

    def test_propagates_annex_get_failure(self, tmp_path: Path) -> None:
        """A CommandError from annex.get() bubbles up so the chain falls through."""
        annex = MagicMock()
        annex.get.side_effect = RuntimeError("annex get failed: no remote knows")
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)

        # The chain wraps the underlying error in IOError with __cause__.
        with pytest.raises(IOError) as exc_info:
            adapter.open("doc.md")
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "annex get failed" in str(exc_info.value.__cause__)

    def test_chain_falls_through_to_next_backend_on_failure(
        self, tmp_path: Path
    ) -> None:
        """When annex-get raises, the next backend is tried."""
        from io import BytesIO

        from datalad_fuse.backends import Backend

        class _StubOK(Backend):
            name = "stub"

            def can_handle(self, key, mode):  # noqa: U100
                return True

            def open_url(self, url, mode="rb", **_kw):  # noqa: U100
                return BytesIO(b"fallback content")

        annex = MagicMock()
        annex.get.side_effect = RuntimeError("nope")
        get = AnnexGetBackend()
        stub = _StubOK()
        adapter = _make_dataset_adapter(
            tmp_path, [get, stub], ["http://example.com/x"], annex=annex
        )

        with adapter.open("doc.md") as f:
            assert f.read() == b"fallback content"


@pytest.mark.ai_generated
class TestAnnexGetRegistration:
    """create_backends() recognises 'annex-get' alongside fsspec/remfile."""

    def test_annex_get_only(self, tmp_path: Path) -> None:
        backends = create_backends("annex-get", tmp_path, caching=False)
        assert len(backends) == 1
        assert backends[0].name == "annex-get"
        assert isinstance(backends[0], AnnexGetBackend)

    def test_annex_get_then_fsspec(self, tmp_path: Path) -> None:
        backends = create_backends("annex-get,fsspec", tmp_path, caching=False)
        assert [b.name for b in backends] == ["annex-get", "fsspec"]

    def test_unknown_still_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backends("nosuch-backend", tmp_path, caching=False)
