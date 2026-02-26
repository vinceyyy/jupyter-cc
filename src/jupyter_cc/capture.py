"""Automatic image capture from IPython display() calls."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

logger = logging.getLogger(__name__)

_IMAGE_FORMATS = frozenset({"image/png", "image/jpeg", "image/jpg", "image/svg+xml"})
_MAX_IMAGES = 20


class ImageCollector:
    """Intercepts images from display() calls by wrapping DisplayPublisher.publish."""

    def __init__(self, shell: InteractiveShell) -> None:
        self._shell = shell
        self._images: list[dict[str, Any]] = []
        self._original_publish: Any = None

    def install(self) -> None:
        """Wrap shell.display_pub.publish to intercept images."""
        self._original_publish = self._shell.display_pub.publish

        @functools.wraps(self._original_publish)
        def _capturing_publish(
            data: dict[str, Any] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            if data:
                for fmt in _IMAGE_FORMATS:
                    if fmt in data:
                        self._images.append(
                            {
                                "format": fmt,
                                "data": data[fmt],
                                "metadata": metadata or {},
                            }
                        )
                        if len(self._images) > _MAX_IMAGES:
                            self._images = self._images[-_MAX_IMAGES:]
            return self._original_publish(data=data, metadata=metadata, **kwargs)

        self._shell.display_pub.publish = _capturing_publish

    def uninstall(self) -> None:
        """Restore original publish method."""
        if self._original_publish is not None:
            self._shell.display_pub.publish = self._original_publish
            self._original_publish = None

    def drain(self) -> list[dict[str, Any]]:
        """Return all captured images and clear the buffer."""
        images = self._images
        self._images = []
        return images

    def format_summary(self, images: list[dict[str, Any]]) -> str:
        """Create a text summary of captured images."""
        if not images:
            return ""
        lines = [f"Captured {len(images)} image(s) from cell execution:"]
        for i, img in enumerate(images, 1):
            lines.append(f"  {i}. {img['format']}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Deprecated stubs — kept temporarily so magics.py imports don't break.
# Task 5 will update magics.py to use ImageCollector and remove these.
# ------------------------------------------------------------------


def extract_images_from_captured(captured_output: Any) -> list[dict[str, Any]]:  # noqa: ARG001
    """Deprecated: use ImageCollector instead."""
    return []


def format_images_summary(images: list[dict[str, Any]]) -> str:  # noqa: ARG001
    """Deprecated: use ImageCollector.format_summary instead."""
    return ""
