"""Depth-weak forward poses: temporal geometry, no finger or fixed-order input."""
import json
from pathlib import Path
import unittest

import numpy as np

from balanced_pointing_rules import BalancedPointing
from realtime_rtmw3d import GestureTally
from test_current_body_rules import body_pose


SEGMENTS = json.loads((Path(__file__).parent /
                      'regression_fixtures/compact_forward_v8.json').read_text(encoding='utf8'))['segments']


def replay(name, offset=0., missing_head=False, fingers=False):
    counter = BalancedPointing(); tally = GestureTally(); events = []
    for row in SEGMENTS['calibration'] + SEGMENTS[name]:
        p, q, s = [np.array(row[k], dtype=float) for k in ('xyz', 'xy', 'scores')]
        if missing_head:
            p[:5] = np.nan; q[:5] = np.nan; s[:5] = 0
        if fingers:
            p = np.concatenate((p, np.full((116,3), np.nan)))
            q = np.concatenate((q, np.full((116,2), 9999.)))
            s = np.concatenate((s, np.ones(116)))
        state, arms = counter.update(1, row['frame'], row['time_s']+offset, p, q, s)
        event = tally.observe(1, counter.people[1]['phase'])
        if event:events.append((event['direction'], event['time_s']-offset))
        json.dumps({'state':state, 'arms':arms})
    return tally.decorate(state), events


def low_side(direction):
    p, _, s = body_pose()
    p[[8,10]] = [[-.40,.24,0],[-.65,.30,0]] if direction=='right' else [[-.05,.24,0],[.25,.30,0]]
    return p, p[:,:2]*200+500, s


def compact_pose():
    p, _, s = body_pose()
    p[[8,10]] = [[-.21,.20,0],[-.125,.11,0]]
    return p, p[:,:2]*200+500, s


class CompactForwardV8Tests(unittest.TestCase):
    def test_first_left_is_not_lost_after_body_reference_settles(self):
        state, events = replay('first_three')
        self.assertEqual([d for d,_ in events], ['left','right','forward'])
        self.assertTrue(state['passed'])
        self.assertTrue(5.2 < events[0][1] < 5.6)

    def test_flat_depth_front_is_confirmed_after_real_side_motion(self):
        state, events = replay('compact_three')
        self.assertEqual(state['counts'], {'forward':1,'left':1,'right':1})
        self.assertEqual([d for d,_ in events], ['left','right','forward'])
        self.assertTrue(10.6 < events[-1][1] < 11.)

    def test_small_front_jitter_does_not_add_another_left(self):
        state, events = replay('jitter_front')
        self.assertEqual(state['counts'], {'forward':1,'left':1,'right':1})
        self.assertEqual([d for d,_ in events], ['left','right','forward'])

    def test_cross_chest_transition_is_not_premature_forward(self):
        state, events = replay('cross_chest_transition')
        self.assertEqual([d for d,_ in events], ['left','right','forward'])
        self.assertTrue(events[-1][1] > 52.25)
        self.assertEqual(state['counts']['forward'], 1)

    def test_initial_compact_side_is_not_guessed_forward(self):
        state, events = replay('initial_compact_side')
        self.assertNotIn('forward', [d for d,_ in events])
        self.assertFalse(state['passed'])

    def test_diagonal_right_is_not_delayed_into_the_front_window(self):
        _, events = replay('diagonal_right')
        right = [t for d,t in events if d=='right']
        self.assertEqual(len(right), 1)
        self.assertTrue(26.1 < right[0] < 26.6)

    def test_missing_head_does_not_disable_compact_forward(self):
        state, events = replay('compact_three', missing_head=True)
        self.assertEqual([d for d,_ in events], ['left','right','forward'])
        self.assertTrue(state['passed'])

    def test_extra_hand_and_finger_points_cannot_change_events(self):
        for name in ('compact_three','jitter_front','cross_chest_transition'):
            with self.subTest(name=name):
                self.assertEqual(replay(name), replay(name, fingers=True))

    def test_video_time_offset_does_not_change_confirmations(self):
        original, events = replay('compact_three')
        shifted, shifted_events = replay('compact_three', offset=100.)
        self.assertEqual(original['counts'], shifted['counts'])
        self.assertEqual([d for d,_ in events], [d for d,_ in shifted_events])
        np.testing.assert_allclose([t for _,t in events], [t for _,t in shifted_events], atol=1e-8)

    def feed(self, counter, tally, data, start, count):
        for frame in range(start, start+count):
            state, _ = counter.update(1, frame, frame/30, *data)
            tally.observe(1, counter.people[1]['phase'])
        return tally.decorate(state)

    def test_compact_forward_accepts_either_preceding_side(self):
        for direction in ('left','right'):
            with self.subTest(direction=direction):
                c=BalancedPointing();t=GestureTally()
                self.feed(c,t,body_pose(),0,20)
                self.feed(c,t,low_side(direction),20,20)
                state=self.feed(c,t,compact_pose(),40,20)
                self.assertEqual(state['done'], [direction,'forward'])
                self.assertFalse(state['passed'])

    def test_compact_chest_pose_without_pointing_motion_does_not_count(self):
        c=BalancedPointing();t=GestureTally()
        self.feed(c,t,body_pose(),0,20)
        state=self.feed(c,t,compact_pose(),20,60)
        self.assertEqual(state['done'], [])

    def test_brief_compact_transition_cannot_increment_forward(self):
        c=BalancedPointing();t=GestureTally()
        self.feed(c,t,body_pose(),0,20)
        self.feed(c,t,low_side('right'),20,20)
        self.feed(c,t,compact_pose(),40,3)
        state=self.feed(c,t,body_pose(),43,20)
        self.assertEqual(state['counts'], {'forward':0,'left':0,'right':1})


if __name__ == '__main__':
    unittest.main()
