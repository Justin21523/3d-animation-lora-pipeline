"""
Placeholder tracker interface for future extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Track:
    track_id: str
    frame_ids: List[str]
    video_id: str
    score: float = 1.0
    meta: Dict = None


def simple_tracks_from_detections(detections: List[Dict]) -> List[Track]:
    """
    Build Track objects grouped by track_id.
    """
    grouped: Dict[str, List[Dict]] = {}
    for det in detections:
        track_id = det.get("track_id")
        if track_id is None:
            continue
        grouped.setdefault(track_id, []).append(det)

    tracks: List[Track] = []
    for track_id, dets in grouped.items():
        frame_ids = [d["frame_id"] for d in dets]
        video_id = dets[0].get("video_id", "")
        tracks.append(Track(track_id=track_id, frame_ids=frame_ids, video_id=video_id, meta={}))
    return tracks

