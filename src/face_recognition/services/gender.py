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
    def __init__(self, vote_window: int = 5):
        self.vote_window = max(1, int(vote_window))
        self._cache: dict[int, deque[tuple[str, str]]] = {}

    def put(self, track_id: int, result: tuple[str, str]) -> None:
        dq = self._cache.setdefault(track_id, deque())
        dq.append((result[0], result[1]))
        while len(dq) > self.vote_window:
            dq.popleft()

    def get(self, track_id: int) -> tuple[str, str] | None:
        hist = self._cache.get(track_id)
        if not hist:
            return None
        votes: OrderedDict[str, list] = OrderedDict()  # label -> [count, confTotal]
        for label, pct in hist:
            if not label or label == "Unknown":
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
            latest = hist[-1]
            return (latest[0], latest[1])
        best, (best_n, best_total) = None, (-1, -1.0)
        for label, (n, total) in votes.items():
            avg = total / n
            best_avg = best_total / best_n if best_n > 0 else -1
            if n > best_n or (n == best_n and avg > best_avg):
                best, best_n, best_total = label, n, total
        return (best, f"{round(best_total / best_n)}%")

    def contains(self, track_id: int) -> bool:
        return track_id in self._cache and len(self._cache[track_id]) > 0

    def retain_all(self, active_ids: set[int]) -> None:
        for k in list(self._cache.keys()):
            if k not in active_ids:
                del self._cache[k]

    def clear(self) -> None: self._cache.clear()
    def size(self) -> int: return len(self._cache)
