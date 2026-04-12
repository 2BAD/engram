"""Input data model for dataset examples."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

# Media types that both Anthropic and OpenAI accept as image inputs.
_IMAGE_MEDIA_TYPES = frozenset(
    {
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
    }
)

# PDF is supported by Anthropic (as document type) and OpenAI (as file input).
_DOCUMENT_MEDIA_TYPES = frozenset(
    {
        'application/pdf',
    }
)

BINARY_MEDIA_TYPES = _IMAGE_MEDIA_TYPES | _DOCUMENT_MEDIA_TYPES


@dataclass
class InputData:
    """A single input example, either text or binary (image/PDF)."""

    filename: str
    text: str | None = None
    data: bytes | None = None
    media_type: str | None = None

    @property
    def is_binary(self) -> bool:
        return self.data is not None

    @property
    def is_image(self) -> bool:
        return self.media_type in _IMAGE_MEDIA_TYPES

    @property
    def is_document(self) -> bool:
        return self.media_type in _DOCUMENT_MEDIA_TYPES

    @property
    def data_base64(self) -> str:
        """Base64-encode the binary data for API payloads."""
        if self.data is None:
            return ''
        return base64.standard_b64encode(self.data).decode('ascii')

    @property
    def text_for_display(self) -> str:
        """Human-readable summary for display/logging."""
        if self.text is not None:
            return self.text
        size = len(self.data) if self.data else 0
        return f'[binary {self.media_type}, {size} bytes]'


def detect_media_type(path: Path) -> str | None:
    """Return a known binary media type for the file, or None for text."""
    mt, _ = mimetypes.guess_type(path.name)
    if mt and mt in BINARY_MEDIA_TYPES:
        return mt
    return None
