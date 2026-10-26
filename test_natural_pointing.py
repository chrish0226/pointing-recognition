"""Natural gestures and real non-pointing activities; no GPU is needed."""
import json
from pathlib import Path
import unittest

import numpy as np

from balanced_pointing_rules import BalancedPointing
from realtime_rtmw3d import GestureTally
from test_current_body_rules import body_pose


SEGMENTS = json.loads((Path(__file__).parent /
                      'regression_fixtures/natural_gestures_real.json').read_text(encoding='utf-8'))['segments']


def replay(name, offset=0):
    counter = BalancedPointing()
    tally = GestureTally()
    events = []
    for row in SEGMENTS[name]:
        p = np.zeros((133, 3)); q = np.zeros((133, 2)); s = np.zeros(133)
        p[:17] = row['xyz']; q[:17] = row['xy']; s[:17] = row['scores']
        q[91:133] = row['hands_xy']; s[91:133] = row['hands_scores']
        state, _ = counter.update(1, row['frame'], row['time_s'] + offset, p, q, s)
        event = tally.observe(1, counter.people[1]['phase'])
        if event:
            events.append((event['direction'], event['time_s'] - offset))
    return tally.decorate(state), events


def compact_pose(direction, pointer=True):
    p, _, s = body_pose()
    if direction == 'left':
        p[[8, 10]] = [[-.30, .20, -.18], [-.18, .24, -.36]]
        sign = 1.
    else:
        p[[8, 10]] = [[-.10, .20, -.18], [-.22, .24, -.36]]
        sign = -1.
    q = p[:, :2] * 200 + 500
    xyz = np.zeros((133, 3)); xy = np.zeros((133, 2)); scores = np.zeros(133)
    xyz[:17] = p; xy[:17] = q; scores[:17] = s
    if pointer:
        xy[117:121] = q[10] + np.array([[0, 0], [3 * sign, 0], [7 * sign, 0], [11 * sign, 0]])
        scores[117:121] = .9
    return xyz, xy, scores


class NaturalPointingTests(unittest.TestCase):
    def feed_compact(self, direction, pointer=True, config=None, missing_head=False):
        counter = BalancedPointing(config)
        tally = GestureTally()
        for frame in range(55):
            arrays = body_pose() if frame < 20 else compact_pose(direction, pointer)
            p, q, s = [v.copy() for v in arrays]
            if missing_head:
                p[:5] = np.nan; q[:5] = np.nan; s[:5] = 0
            state, _ = counter.update(1, frame, frame / 30, p, q, s)
            tally.observe(1, counter.people[1]['phase'])
        return tally.decorate(state)

    def test_real_low_bent_gestures_follow_actual_left_right_front_order(self):
        state, events = replay('natural_three')
        self.assertEqual(state['counts'], {'forward': 1, 'left': 1, 'right': 1})
        self.assertEqual([d for d, _ in events], ['left', 'right', 'forward'])
        for (_, t), (lo, hi) in zip(events, ((2.6, 2.95), (3., 3.5), (3.55, 3.95))):
            self.assertTrue(lo <= t <= hi, events)

    def test_gestures_do_not_depend_on_absolute_video_time(self):
        original, _ = replay('natural_three')
        shifted, events = replay('natural_three', offset=100.)
        self.assertEqual(original['counts'], shifted['counts'])
        self.assertEqual([d for d, _ in events], ['left', 'right', 'forward'])

    def test_real_glove_handling_does_not_count(self):
        state, events = replay('glove_handling')
        self.assertEqual(state['counts'], dict.fromkeys(('forward', 'left', 'right'), 0))
        self.assertEqual(events, [])

    def test_real_walking_swing_does_not_count(self):
        state, events = replay('walking_swing')
        self.assertEqual(state['done'], [])
        self.assertEqual(events, [])

    def test_compact_cross_chest_point_uses_agreeing_forearm_and_index(self):
        for direction in ('left', 'right'):
            with self.subTest(direction=direction):
                state = self.feed_compact(direction)
                self.assertEqual(state['done'], [direction])
                self.assertEqual(state['counts'][direction], 1)
                self.assertFalse(state['passed'])

    def test_optional_hands_missing_preserves_body_fallback(self):
        natural = self.feed_compact('left', pointer=False)
        body_only = self.feed_compact('left', pointer=False, config={'natural_gestures': False})
        self.assertEqual(natural['counts'], body_only['counts'])

    def test_compact_side_does_not_require_head_landmarks(self):
        state = self.feed_compact('left', missing_head=True)
        self.assertEqual(state['done'], ['left'])

    def test_full_body_and_hand_inputs_are_not_changed(self):
        arrays = compact_pose('left')
        copies = [v.copy() for v in arrays]
        counter = BalancedPointing()
        for frame in range(30):
            state, arms = counter.update(1, frame, frame / 30, *arrays)
            json.dumps({'state': state, 'arms': arms})
        for current, original in zip(arrays, copies):
            np.testing.assert_array_equal(current, original)

    def test_unusable_finger_points_do_not_raise_or_disable_body_rules(self):
        for invalid in ('nan', 'low_confidence', 'detached'):
            with self.subTest(invalid=invalid):
                counter = BalancedPointing()
                for frame in range(40):
                    p, q, s = compact_pose('left')
                    if invalid == 'nan':
                        q[117:121] = np.nan
                    elif invalid == 'low_confidence':
                        s[117:121] = .1
                    else:
                        q[117:121] += 500
                    state, arms = counter.update(1, frame, frame / 30, p, q, s)
                self.assertEqual(state['done'], ['forward'])


if __name__ == '__main__':
    unittest.main()
