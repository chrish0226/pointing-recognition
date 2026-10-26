"""Program regressions: occlusion isolation, time voting and known walking points."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from balanced_pointing_rules import BalancedPointing
from realtime_rtmw3d import GestureTally
from test_current_body_rules import body_pose
from test_side_forward_rules import FIXTURE


SEGMENTS = json.loads((Path(__file__).parent /
                      'regression_fixtures/body_direction_v7.json').read_text(encoding='utf8'))['segments']


def replay(rows):
    counter = BalancedPointing()
    tally = GestureTally()
    events = []
    for row in rows:
        state, _ = counter.update(1, row['frame'], row['time_s'], row['xyz'], row['xy'], row['scores'])
        event = tally.observe(1, counter.people[1]['phase'])
        if event:
            events.append((event['direction'], event['time_s']))
    return tally.decorate(state), events


class BodyDirectionV7Tests(unittest.TestCase):
    def test_walking_left_is_not_blocked_and_front_does_not_fill_right(self):
        state, events = replay(SEGMENTS['4'])
        self.assertEqual(state['counts'], {'forward': 1, 'left': 1, 'right': 0})
        self.assertEqual([d for d, _ in events], ['left', 'forward'])
        self.assertTrue(1.7 < events[0][1] < 2.05)

    def test_walking_low_front_is_not_suppressed(self):
        state, events = replay(SEGMENTS['13'])
        self.assertEqual([d for d, _ in events], ['left', 'forward'])
        self.assertTrue(2.0 < events[-1][1] < 2.3)
        self.assertFalse(state['passed'])

    def test_agreeing_body_references_preserve_left(self):
        state, events = replay(FIXTURE['calibration'] + SEGMENTS['left_after_turn'])
        self.assertEqual(state['counts'], {'forward': 0, 'left': 1, 'right': 0})
        self.assertTrue(28.4 < events[0][1] < 29.)

    def test_agreeing_body_references_preserve_right(self):
        state, events = replay(FIXTURE['calibration'] + SEGMENTS['right_after_turn'])
        self.assertEqual(state['counts'], {'forward': 0, 'left': 0, 'right': 1})
        self.assertTrue(43. < events[0][1] < 43.6)

    def run_occluded(self, direction, invalid_ids):
        counter = BalancedPointing()
        tally = GestureTally()
        for frame in range(50):
            p, q, s = body_pose(direction if frame >= 20 else None)
            p[invalid_ids] = np.nan; q[invalid_ids] = np.nan; s[invalid_ids] = 0
            state, _ = counter.update(1, frame, frame / 30, p, q, s)
            tally.observe(1, counter.people[1]['phase'])
        return tally.decorate(state)

    def test_occluded_other_arm_does_not_disable_visible_gesture(self):
        for direction in ('left', 'right', 'forward'):
            with self.subTest(direction=direction):
                state = self.run_occluded(direction, [8, 10] if direction=='left' else [7, 9])
                self.assertEqual(state['done'], [direction])
                self.assertEqual(state['counts'][direction], 1)

    def test_invalid_active_arm_never_creates_a_gesture(self):
        self.assertEqual(self.run_occluded('right', [8, 10])['done'], [])

    def test_missing_torso_reference_never_creates_a_gesture(self):
        self.assertEqual(self.run_occluded('left', [11])['done'], [])

    def test_high_frame_rate_burst_must_have_sufficient_duration(self):
        counter = BalancedPointing()
        for frame in range(24):
            counter.update(1, frame, frame / 120, *body_pose())
        for frame in range(24, 30):
            state, _ = counter.update(1, frame, frame / 120, *body_pose('left'))
        self.assertEqual(state['done'], [])
        for frame in range(30, 50):
            state, _ = counter.update(1, frame, frame / 120, *body_pose('left'))
        self.assertEqual(state['done'], ['left'])

    def sample_vote(self):
        counter = BalancedPointing()
        for frame in range(25):
            counter.update(1, frame, frame / 30, *body_pose('forward' if frame >= 20 else None))
        return next(deepcopy(v) for v in reversed(counter.people[1]['votes']) if v['direction'])

    def test_expired_votes_cannot_combine_into_a_new_confirmation(self):
        counter = BalancedPointing()
        vote = self.sample_vote()
        fresh = [{**vote, 'frame': i, 'time_s': t} for i, t in enumerate((0., .04, .08, .12))]
        self.assertIsNotNone(counter._stable_candidate(fresh))
        stale = [{**vote, 'frame': i, 'time_s': t} for i, t in enumerate((0., .10, .20, .55))]
        self.assertIsNone(counter._stable_candidate(stale))

    def test_short_confirmation_requires_consecutive_spatial_support(self):
        counter = BalancedPointing()
        vote = {**self.sample_vote(), 'spatial_conflict': True, 'margin': .5, 'quality': .9}
        consecutive = [{**vote, 'frame': i, 'time_s': t} for i, t in enumerate((0., .04, .08))]
        phase = {'direction': 'left'}
        self.assertIsNotNone(counter._stable_candidate(consecutive, phase))
        interrupted = [consecutive[0], {'direction': None, 'time_s': .02},
                       consecutive[1], {'direction': None, 'time_s': .06}, consecutive[2]]
        self.assertIsNone(counter._stable_candidate(interrupted, phase))


if __name__ == '__main__':
    unittest.main()
