"""GenderClassifier + GenderCacheManager — port of Java versions."""
from __future__ import annotations

import math
from collections import OrderedDict, deque


def classify(male: float, female: float, min_conf: float) -> tuple[str, str]:
    if (not math.isfinite(male) or not math.isfinite(female) or male < 0 or female < 0
            or not math.isfinite(min_conf) or min_conf <= 0):
        return ("Unknown", "")
    total = male + female
    if total <= 0:
        return ("Unknown", "")
    mp, fp = male / total, female / total
    label = "Male" if mp >= fp else "Female"
    conf = max(mp, fp)
    pct = f"{round(conf * 100)}%"
    if conf < min_conf:
        return ("Unknown", pct)
    return (label, pct)


class GenderCacheManager:
    """Sliding vote window per tracked face.

    A label is only exposed once at least ``min_votes`` usable predictions
    agree on it. One blurred/odd camera frame can no longer flash a wrong
    "Male" on screen — the track simply keeps its previous label (or none)
    until enough clean votes arrive.
    """

    def __init__(self, vote_window: int = 5, min_votes: int = 3):
        self.vote_window = max(1, int(vote_window))
        self.min_votes = max(1, min(int(min_votes), self.vote_window))
        self._cache: dict[int, deque[tuple[str, str]]] = {}

    def put(self, track_id: int, result: tuple[str, str]) -> None:
        dq = self._cache.setdefault(track_id, deque())
        dq.append((result[0], result[1]))
        while len(dq) > self.vote_window:
            dq.popleft()

    def votes(self, track_id: int) -> int:
        """Number of usable (non-Unknown) predictions for this track."""
        hist = self._cache.get(track_id)
        if not hist:
            return 0
        return sum(1 for label, _ in hist if label and label != "Unknown")

    def get(self, track_id: int) -> tuple[str, str] | None:
        """Return the agreed label, or None while the vote is undecided."""
        hist = self._cache.get(track_id)
        if not hist:
            return None
        votes: OrderedDict[str, list] = OrderedDict()  # label -> [count, confTotal]
        unknown = 0
        for label, pct in hist:
            if not label or label == "Unknown":
                unknown += 1
                continue
            try:
                conf = int(str(pct).replace("%", "").strip() or 0)
            except ValueError:
                conf = 0
            if label in votes:
                votes[label][0] += 1
                votes[label][1] += conf
            else:
                votes[label] = [1, conf]
        if not votes:
            # Only rejected/low-confidence crops so far: stay quiet briefly,
            # then report Unknown instead of guessing a binary label.
            return ("Unknown", "") if unknown >= self.min_votes else None
        best, (best_n, best_total) = None, (-1, -1.0)
        for label, (n, total) in votes.items():
            avg = total / n
            best_avg = best_total / best_n if best_n > 0 else -1
            if n > best_n or (n == best_n and avg > best_avg):
                best, best_n, best_total = label, n, total
        if best_n < self.min_votes:
            return None
        return (best, f"{round(best_total / best_n)}%")

    def contains(self, track_id: int) -> bool:
        return track_id in self._cache and len(self._cache[track_id]) > 0

    def retain_all(self, active_ids: set[int]) -> None:
        for k in list(self._cache.keys()):
            if k not in active_ids:
                del self._cache[k]

    def clear(self) -> None: self._cache.clear()
    def size(self) -> int: return len(self._cache)
