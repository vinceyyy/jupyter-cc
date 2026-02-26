"""Unit tests for jupyter_cc.capture — ImageCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jupyter_cc.capture import ImageCollector


@pytest.fixture
def mock_shell() -> MagicMock:
    shell = MagicMock()
    shell.display_pub = MagicMock()
    shell.display_pub.publish = MagicMock()
    return shell


@pytest.fixture
def collector(mock_shell: MagicMock) -> ImageCollector:
    return ImageCollector(mock_shell)


class TestImageCollector:
    def test_install_wraps_publish(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        original = mock_shell.display_pub.publish
        collector.install()
        assert mock_shell.display_pub.publish is not original

    def test_uninstall_restores_publish(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        original = mock_shell.display_pub.publish
        collector.install()
        collector.uninstall()
        assert mock_shell.display_pub.publish is original

    def test_captures_png_image(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        # Simulate a display() call with PNG data by calling the wrapper directly
        mock_shell.display_pub.publish(
            data={"image/png": "base64data==", "text/plain": "<Figure>"},
            metadata={},
        )
        images = collector.drain()
        assert len(images) == 1
        assert images[0]["format"] == "image/png"
        assert images[0]["data"] == "base64data=="

    def test_captures_multiple_formats(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        mock_shell.display_pub.publish(
            data={"image/png": "png_data", "image/svg+xml": "<svg/>"},
            metadata={},
        )
        images = collector.drain()
        assert len(images) == 2

    def test_ignores_non_image_data(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        mock_shell.display_pub.publish(
            data={"text/html": "<table>...</table>"},
            metadata={},
        )
        images = collector.drain()
        assert len(images) == 0

    def test_drain_clears_buffer(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        mock_shell.display_pub.publish(
            data={"image/png": "data1"},
            metadata={},
        )
        collector.drain()
        assert collector.drain() == []

    def test_cap_at_20_images(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        for i in range(25):
            mock_shell.display_pub.publish(
                data={"image/png": f"data_{i}"},
                metadata={},
            )
        images = collector.drain()
        assert len(images) == 20
        assert images[0]["data"] == "data_5"
        assert images[-1]["data"] == "data_24"

    def test_captures_cell_execution_number(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        mock_shell.execution_count = 5
        collector.install()
        mock_shell.display_pub.publish(data={"image/png": "data"}, metadata={})
        images = collector.drain()
        assert images[0]["cell"] == 5

    def test_format_summary_with_cells(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        mock_shell.execution_count = 3
        collector.install()
        mock_shell.display_pub.publish(data={"image/png": "d1"}, metadata={})
        mock_shell.execution_count = 5
        mock_shell.display_pub.publish(data={"image/png": "d2"}, metadata={})
        images = collector.drain()
        summary = collector.format_summary(images)
        assert "Captured 2 image(s) from cell execution [3, 5]" == summary

    def test_format_summary_empty(self, collector: ImageCollector) -> None:
        assert collector.format_summary([]) == ""

    def test_passthrough_to_original(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        original = mock_shell.display_pub.publish
        collector.install()
        mock_shell.display_pub.publish(
            data={"image/png": "data"},
            metadata={"isolated": True},
        )
        original.assert_called_once_with(
            data={"image/png": "data"},
            metadata={"isolated": True},
        )
