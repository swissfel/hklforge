"""A small CIF reader: data blocks with their items and loops, as strings.

Enough for crystal structures as published (cell, space group, symmetry
operations, atom sites, displacement parameters): quoted values, ;-text
fields, comments and loops. Numbers keep their text, including standard
uncertainties; ``number()`` reads them ("0.4008(2)" -> 0.4008).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+""")


@dataclass
class Block:
    name: str
    items: dict = field(default_factory=dict)          # tag -> value
    loops: list = field(default_factory=list)          # [{tag: [values]}]

    def get(self, *tags, default=None):
        """The first of ``tags`` present (CIF 1 and 2 spellings), as text."""
        for tag in tags:
            if tag.lower() in self.items:
                return self.items[tag.lower()]
        return default

    def loop(self, *tags) -> dict | None:
        """The loop containing the first of ``tags`` found, {tag: [values]}."""
        for tag in tags:
            for lp in self.loops:
                if tag.lower() in lp:
                    return lp
        return None


def number(text, default=None):
    """'0.4008(2)' -> 0.4008; '.' and '?' (CIF's unknown) -> default."""
    if text is None:
        return default
    text = str(text).strip()
    if text in (".", "?", ""):
        return default
    return float(re.sub(r"\(\d+\)$", "", text))


def _tokens(text: str):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):                       # text field up to the next line starting with ;
            parts = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                parts.append(lines[i])
                i += 1
            i += 1
            yield "\n".join(parts).strip()
            continue
        for m in _TOKEN.finditer(line):
            tok = m.group(0)
            if tok.startswith("#"):
                break
            if tok[0] in "'\"" and tok[-1] == tok[0] and len(tok) > 1:
                tok = tok[1:-1]
            yield tok
        i += 1


def parse(text: str) -> list[Block]:
    """All data blocks of a CIF text."""
    blocks, block = [], None
    toks = list(_tokens(text))
    i = 0
    while i < len(toks):
        tok = toks[i]
        low = tok.lower()
        if low.startswith("data_"):
            block = Block(tok[5:])
            blocks.append(block)
            i += 1
        elif block is None:
            i += 1
        elif low == "loop_":
            i += 1
            tags = []
            while i < len(toks) and toks[i].startswith("_"):
                tags.append(toks[i].lower())
                i += 1
            values = []
            while i < len(toks) and not (toks[i].startswith("_") or toks[i].lower() in ("loop_",)
                                         or toks[i].lower().startswith(("data_", "save_"))):
                values.append(toks[i])
                i += 1
            if tags:
                block.loops.append({t: values[j::len(tags)] for j, t in enumerate(tags)})
        elif tok.startswith("_"):
            block.items[low] = toks[i + 1] if i + 1 < len(toks) else ""
            i += 2
        else:
            i += 1
    return blocks
