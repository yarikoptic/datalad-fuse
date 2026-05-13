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


@pytest.mark.ai_generated
class TestAnnexGetDropKwargValidation:
    """Constructor rejects invalid drop / drop_mode values."""

    @pytest.mark.parametrize("bad", ["", "yes", "obtain", "ALL", "DROP-ALL"])
    def test_invalid_drop_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="drop="):
            AnnexGetBackend(drop=bad)

    @pytest.mark.parametrize("bad", ["", "slow", "FAST", "f", "--force"])
    def test_invalid_drop_mode_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="drop_mode="):
            AnnexGetBackend(drop_mode=bad)

    @pytest.mark.parametrize("drop", ["none", "obtained", "all"])
    @pytest.mark.parametrize("mode", ["regular", "fast", "force"])
    def test_valid_combinations(self, drop: str, mode: str) -> None:
        # Should not raise.
        backend = AnnexGetBackend(drop=drop, drop_mode=mode)
        assert backend._drop == drop
        assert backend._drop_mode == mode


@pytest.mark.ai_generated
class TestAnnexGetTracking:
    """`open` populates `_fetched` only on successful materialisation."""

    def test_successful_fetch_recorded(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"a")
        (tmp_path / "b.bin").write_bytes(b"b")
        annex = MagicMock()
        annex.get = MagicMock(return_value=None)
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)

        with adapter.open("a.bin"):
            pass
        with adapter.open("b.bin"):
            pass

        assert backend._fetched == {"a.bin", "b.bin"}

    def test_failed_materialisation_not_recorded(self, tmp_path: Path) -> None:
        """If post-get the file isn't there, the relpath stays unrecorded."""
        annex = MagicMock()
        annex.get = MagicMock(return_value=None)
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)

        with pytest.raises(IOError):
            adapter.open("missing.bin")
        assert backend._fetched == set()

    def test_failed_annex_get_not_recorded(self, tmp_path: Path) -> None:
        annex = MagicMock()
        annex.get.side_effect = RuntimeError("nope")
        backend = AnnexGetBackend()
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)

        with pytest.raises(IOError):
            adapter.open("x.bin")
        assert backend._fetched == set()


@pytest.mark.ai_generated
class TestAnnexGetCloseDrop:
    """`close(adapter)` performs the configured drop."""

    def _setup(self, tmp_path: Path, drop: str, drop_mode: str):
        (tmp_path / "a.bin").write_bytes(b"a")
        (tmp_path / "b.bin").write_bytes(b"b")
        annex = MagicMock()
        annex.get = MagicMock(return_value=None)
        annex.call_annex = MagicMock(return_value=None)
        backend = AnnexGetBackend(drop=drop, drop_mode=drop_mode)
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=annex)
        return backend, adapter, annex

    def test_drop_none_is_noop(self, tmp_path: Path) -> None:
        backend, adapter, annex = self._setup(tmp_path, "none", "regular")
        with adapter.open("a.bin"):
            pass
        adapter.close()
        annex.call_annex.assert_not_called()

    @pytest.mark.parametrize(
        "mode,expected_flags",
        [("regular", []), ("fast", ["--fast"]), ("force", ["--force"])],
    )
    def test_drop_obtained_with_modes(
        self, tmp_path: Path, mode: str, expected_flags: list
    ) -> None:
        backend, adapter, annex = self._setup(tmp_path, "obtained", mode)
        with adapter.open("a.bin"):
            pass
        with adapter.open("b.bin"):
            pass
        adapter.close()

        annex.call_annex.assert_called_once_with(
            ["drop", *expected_flags], files=["a.bin", "b.bin"]
        )

    def test_drop_obtained_empty_set_is_noop(self, tmp_path: Path) -> None:
        """Nothing was fetched → no drop call is issued."""
        backend, adapter, annex = self._setup(tmp_path, "obtained", "regular")
        # Don't open anything before close.
        adapter.close()
        annex.call_annex.assert_not_called()

    @pytest.mark.parametrize(
        "mode,expected_flags",
        [("regular", []), ("fast", ["--fast"]), ("force", ["--force"])],
    )
    def test_drop_all_with_modes(
        self, tmp_path: Path, mode: str, expected_flags: list
    ) -> None:
        backend, adapter, annex = self._setup(tmp_path, "all", mode)
        with adapter.open("a.bin"):
            pass
        adapter.close()

        annex.call_annex.assert_called_once_with(["drop", "--all", *expected_flags])

    def test_double_close_is_idempotent(self, tmp_path: Path) -> None:
        """Calling close() twice does not re-issue the drop."""
        backend, adapter, annex = self._setup(tmp_path, "obtained", "regular")
        with adapter.open("a.bin"):
            pass
        adapter.close()
        adapter.close()
        annex.call_annex.assert_called_once()

    def test_drop_skipped_when_no_annex(self, tmp_path: Path) -> None:
        """A plain git repo (annex=None) can't be dropped — silently skip."""
        backend = AnnexGetBackend(drop="all", drop_mode="force")
        backend._fetched.add("a.bin")  # pretend we somehow tracked it
        adapter = _make_dataset_adapter(tmp_path, [backend], [], annex=None)
        # Should not raise even though annex is None.
        adapter.close()

    def test_close_exception_is_logged_not_raised(self, tmp_path: Path, caplog) -> None:
        """A failing backend close() doesn't propagate out of adapter.close."""
        import logging as _logging

        caplog.set_level(_logging.WARNING, logger="datalad.fuse.adapter")
        backend, adapter, annex = self._setup(tmp_path, "obtained", "regular")
        annex.call_annex.side_effect = RuntimeError("drop blew up")
        with adapter.open("a.bin"):
            pass
        adapter.close()  # must not raise
        assert any("annex-get close() failed" in r.message for r in caplog.records)


@pytest.mark.ai_generated
class TestAnnexGetConfigIntegration:
    """`create_backends` reads datalad config keys for drop semantics."""

    def test_config_drives_drop(self, tmp_path: Path, monkeypatch) -> None:
        from datalad import cfg

        # datalad's `cfg` is a runtime-mutable ConfigManager; patch its `get`
        # to inject the values the production code would read.
        def fake_get(key: str, default=None):
            return {
                "datalad.fusefs.annex-get.drop": "obtained",
                "datalad.fusefs.annex-get.drop-mode": "fast",
            }.get(key, default)

        monkeypatch.setattr(cfg, "get", fake_get)
        backends = create_backends("annex-get", tmp_path, caching=False)
        assert isinstance(backends[0], AnnexGetBackend)
        assert backends[0]._drop == "obtained"
        assert backends[0]._drop_mode == "fast"

    def test_config_defaults_when_unset(self, tmp_path: Path) -> None:
        backends = create_backends("annex-get", tmp_path, caching=False)
        assert backends[0]._drop == "none"
        assert backends[0]._drop_mode == "regular"
