"""A fetched source document, before chunking."""

import html
import re

from pydantic import BaseModel, Field

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


# MedlinePlus entity-encodes the markup inside its FullSummary field, so one
# strip-then-unescape pass leaves literal tags behind. Repeat until the text
# stops changing; three passes is far more than any real payload needs.
_MAX_CLEAN_PASSES = 3


def clean_text(raw: str) -> str:
    """Strip markup, decode entities, and collapse whitespace.

    Tags are removed before entities are decoded on each pass, so an encoded
    angle bracket in the prose is never mistaken for a tag. Passes repeat because
    decoding can itself reveal markup.
    """
    text = raw
    for _ in range(_MAX_CLEAN_PASSES):
        decoded = html.unescape(_TAG.sub(" ", text))
        if decoded == text:
            break
        text = decoded

    return _WHITESPACE.sub(" ", text).strip()


class Document(BaseModel):
    """One retrieved source document, cleaned but not yet chunked."""

    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str
    text: str = Field(min_length=1)
