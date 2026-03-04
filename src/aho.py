
from __future__ import annotations

from dataclasses import dataclass
from collections import deque


def _is_word_char(c: str) -> bool:
    # Word chars for boundary checks: letters/digits/underscore
    # Hyphen is NOT word char, so "COVID-19" matches as a whole if term includes '-'.
    return c.isalnum() or c == "_"


@dataclass
class _Node:
    nxt: dict[str, int]
    link: int
    out: list[str]

    def __init__(self) -> None:
        self.nxt = {}
        self.link = 0
        self.out = []


class AhoCorasick:
    """
    Simple Aho-Corasick automaton for multi-pattern matching.
    We match case-insensitively by lowering both patterns and text.
    Outputs canonical terms (original cased term provided at build time).
    """

    def __init__(self, terms: list[str]) -> None:
        # Map lower(term) -> canonical term (first seen)
        canon: dict[str, str] = {}
        for t in terms:
            lt = t.lower()
            if lt not in canon:
                canon[lt] = t

        self._canon = canon
        self._nodes: list[_Node] = [_Node()]

        for lt, ct in canon.items():
            self._add_pattern(lt, ct)

        self._build_links()

    def _add_pattern(self, pat_lower: str, canon_term: str) -> None:
        v = 0
        for ch in pat_lower:
            nxt = self._nodes[v].nxt.get(ch)
            if nxt is None:
                nxt = len(self._nodes)
                self._nodes[v].nxt[ch] = nxt
                self._nodes.append(_Node())
            v = nxt
        self._nodes[v].out.append(canon_term)

    def _build_links(self) -> None:
        q = deque()
        for ch, u in self._nodes[0].nxt.items():
            self._nodes[u].link = 0
            q.append(u)

        while q:
            v = q.popleft()
            for ch, u in self._nodes[v].nxt.items():
                q.append(u)
                j = self._nodes[v].link
                while j and ch not in self._nodes[j].nxt:
                    j = self._nodes[j].link
                self._nodes[u].link = self._nodes[j].nxt.get(ch, 0)
                self._nodes[u].out.extend(self._nodes[self._nodes[u].link].out)

    def find(self, text: str) -> list[tuple[int, int, str]]:
        """
        Returns matches as (start, end, term) in original text indices.
        Matching is case-insensitive, but boundaries are checked against original text.
        """
        low = text.lower()
        res: list[tuple[int, int, str]] = []
        v = 0

        for i, ch in enumerate(low):
            while v and ch not in self._nodes[v].nxt:
                v = self._nodes[v].link
            v = self._nodes[v].nxt.get(ch, 0)

            if self._nodes[v].out:
                for term in self._nodes[v].out:
                    L = len(term)
                    end = i + 1
                    start = end - L
                    if start < 0:
                        continue
                    # Word boundary check (against original text)
                    left_ok = True
                    right_ok = True
                    if start - 1 >= 0 and _is_word_char(text[start - 1]):
                        left_ok = False
                    if end < len(text) and _is_word_char(text[end]):
                        right_ok = False
                    if left_ok and right_ok:
                        res.append((start, end, term))

        # Sort by start; if same start, longer first
        res.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        return res
