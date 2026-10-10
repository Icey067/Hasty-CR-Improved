"""Perception flicker must not invent purchases or corrupt learning labels."""
from types import SimpleNamespace

from tests.test_brain import make_state
from brain.arena import Cell
from brain.policy import Brain
from brain.tracker import Tracker, STALE_SECONDS
from scripts.live_perception import configure_unit_detection


def test_cycle_counts_one_purchase_for_multiple_units_and_reacquisition():
    brain = Brain(learn=False, use_advisor=False)
    state = make_state([], 5, enemies=[("minion", 4, 10), ("minion", 5, 10)])
    brain.observe(state, 0, now=100)
    assert brain.opponent.plays == 1
    spent = brain.enemy.spent
    brain.observe(make_state([], 5), 3, now=103)
    brain.observe(state, 4, now=104)
    assert brain.opponent.plays == 1
    assert brain.enemy.spent == spent
    brain.observe(make_state([], 5), 11, now=111)
    brain.observe(state, 12, now=112)
    assert brain.opponent.plays == 2


def test_expired_track_cannot_be_revived_without_an_empty_frame():
    tracker = Tracker()
    first = tracker.update([("knight", Cell(4, 10))], now=100)[0]
    second = tracker.update([("knight", Cell(4, 10))], now=101+STALE_SECONDS)[0]
    assert second is not first
    assert second.hits == 1


def test_short_detection_gap_preserves_identity():
    tracker = Tracker()
    first = tracker.update([("knight", Cell(4, 10))], now=100)[0]
    tracker.update([], now=100.2)
    assert tracker.update([("knight", Cell(4, 11))], now=100.4)[0] is first


def test_learning_pairs_names_with_ids_after_detection_order_changes(tmp_path):
    from brain.experience import ExperienceBook
    from brain.policy import Candidate

    brain = Brain(learn=False, use_advisor=False)
    brain.book = ExperienceBook(tmp_path / "learned.json", tmp_path / "matchups.json")
    rows = [("knight", 4, 20), ("musketeer", 5, 20)]
    brain.observe(make_state([], 5, enemies=rows), 0, now=100)
    brain.last_obs = brain.observe(make_state([], 5, enemies=rows[::-1]), 1, now=101)
    brain.confirm(Candidate("cannon", 0, 4, 22, "defend_cannon", "defend_cannon"), 101)
    episode = brain.book.pending[0]
    assert episode.enemy_units == ["musketeer", "knight"]
    assert [brain.tracker.tracks[i].name for i in episode.enemy_ids] == episode.enemy_units


def test_upstream_units_skipped_only_with_ready_yolo():
    upstream = object()
    detector = SimpleNamespace(unit_detector=upstream)
    configure_unit_detection(detector, None)
    assert detector.unit_detector is upstream
    configure_unit_detection(detector, SimpleNamespace(ready=False))
    assert detector.unit_detector is upstream
    configure_unit_detection(detector, SimpleNamespace(ready=True))
    assert detector.unit_detector.run(None) == ([], [])
