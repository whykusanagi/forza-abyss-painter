"""badge_path() must prefer the new whyKusanagi logo over the legacy
fd6_128.png fallback (user-reported regression: collapsed BrandBanner
showed the OLD upstream FD6 logo)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from forza_abyss_painter.gui import brand_banner


def test_root_badge_resolves(tmp_path):
    """A themed badge at the repo root (e.g. Pink.png) resolves to that path."""
    (tmp_path / "Pink.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with patch.object(brand_banner, "_bundle_root", return_value=tmp_path):
        result = brand_banner.badge_path("Pink.png")
    assert result == tmp_path / "Pink.png"


def test_assets_dir_badge_resolves(tmp_path):
    """The new whyKusanagi logo lives in assets/. badge_path must find it."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "forza_abyss_painter_logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    with patch.object(brand_banner, "_bundle_root", return_value=tmp_path):
        result = brand_banner.badge_path("forza_abyss_painter_logo.png")
    assert result == tmp_path / "assets" / "forza_abyss_painter_logo.png"


def test_assets_preferred_over_legacy_fd6(tmp_path):
    """When BOTH assets/forza_abyss_painter_logo.png AND tools/fd6_128.png
    exist, assets/ must win. This is the actual regression scenario:
    the old tools/fd6_128.png is still in the tree as a holdover and
    was getting picked first."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "forza_abyss_painter_logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "fd6_128.png").write_bytes(b"OLD")
    with patch.object(brand_banner, "_bundle_root", return_value=tmp_path):
        result = brand_banner.badge_path("forza_abyss_painter_logo.png")
    assert result == tmp_path / "assets" / "forza_abyss_painter_logo.png"
    assert "fd6_128" not in str(result)


def test_legacy_fallback_only_when_nothing_else_present(tmp_path):
    """If neither root nor assets/ has the requested file, AND fd6_128.png
    exists, only THEN fall back to fd6_128 (dev-env safety net)."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "fd6_128.png").write_bytes(b"OLD")
    with patch.object(brand_banner, "_bundle_root", return_value=tmp_path):
        result = brand_banner.badge_path("nonexistent.png")
    assert result == tmp_path / "tools" / "fd6_128.png"


def test_returns_none_when_nothing_present(tmp_path):
    with patch.object(brand_banner, "_bundle_root", return_value=tmp_path):
        result = brand_banner.badge_path("nonexistent.png")
    assert result is None
