"""General weak-depth, level-height rotations plus real negative regressions."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from balanced_pointing_rules import BalancedPointing
from realtime_rtmw3d import GestureTally
from test_current_body_rules import body_pose

SEGMENTS = json.loads((Path(__file__).parent /
                      'regression_fixtures/forward_geometry_v9.json').read_text(encoding='utf8'))['segments']


def sample(direction=None, yaw=0, scale=200, arm='R', head=True, offset=500):
    p, _, s = body_pose()
    el, wr = (8, 10) if arm == 'R' else (7, 9)
    shoulder_x = -.2 if arm == 'R' else .2
    if direction in ('left', 'right'):
        p[[el, wr]] = [[shoulder_x, .20, 0],
                       [shoulder_x + (.25 if direction == 'left' else -.25), .15, 0]]
    if direction == 'forward':
        p[[el, wr]] = [[shoulder_x, .20, 0], [shoulder_x, .15, -.25]]
    theta = math.radians(yaw)
    c, t = math.cos(theta), math.sin(theta)
    p = p @ np.array([[c, 0, t], [0, 1, 0], [-t, 0, c]]).T
    q = p[:, :2] * scale + offset
    if direction == 'forward':
        p[[el, wr], 2] = p[6 if arm == 'R' else 5, 2]
    if not head:
        p[:5] = np.nan; q[:5] = np.nan; s[:5] = 0
    return p, q, s


class ForwardGeometryV9Tests(unittest.TestCase):
    def setUp(self):
        self.rules = BalancedPointing()
        self.tally = GestureTally()
        self.frame = 0
        self.events = []

    def feed(self, data, count=1, fps=30, tid=1):
        for _ in range(count):
            arrays = data(self.frame) if callable(data) else data
            state, arms = self.rules.update(tid, self.frame, self.frame / fps, *arrays)
            event = self.tally.observe(tid, self.rules.people[tid]['phase'])
            if event:
                self.events.append((event['direction'], event['time_s']))
            json.dumps({'state': state, 'arms': arms})
            self.frame += 1
        return self.tally.decorate(state)

    def sequence(self, yaw=0, side='left', arm='R', scale=200, head=True, offset=500, fps=30):
        options = dict(yaw=yaw, arm=arm, scale=scale, head=head, offset=offset)
        self.feed(sample(**options), 2 * fps, fps)
        self.feed(sample(side, **options), fps, fps)
        return self.feed(sample('forward', **options), fps, fps)

    def assert_sequence(self, state, side):
        self.assertEqual([d for d, _ in self.events], [side, 'forward'])
        self.assertEqual(state['counts']['forward'], 1)
        self.assertFalse(state['passed'])

    def test_level_height_flat_depth_front_from_either_side(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                self.setUp(); self.assert_sequence(self.sequence(side=side), side)

    def test_moderate_camera_angles_do_not_depend_on_start_side(self):
        for yaw in (-35, -30, -20, 20, 30, 35):
            for side in ('left', 'right'):
                with self.subTest(yaw=yaw, side=side):
                    self.setUp(); self.assert_sequence(self.sequence(yaw=yaw, side=side), side)

    def test_both_arms_use_the_same_geometry(self):
        for arm in ('L', 'R'):
            for side in ('left', 'right'):
                with self.subTest(arm=arm, side=side):
                    self.setUp(); self.assert_sequence(self.sequence(arm=arm, side=side), side)

    def test_image_scale_and_translation_do_not_change_counts(self):
        for scale, offset in ((80, 150), (140, 500), (300, 1500), (640, -900)):
            with self.subTest(scale=scale, offset=offset):
                self.setUp(); self.assert_sequence(self.sequence(scale=scale, offset=offset), 'left')

    def test_head_is_optional_at_moderate_angles(self):
        for yaw in (-30, 0, 30):
            with self.subTest(yaw=yaw):
                self.setUp(); self.assert_sequence(self.sequence(yaw=yaw, head=False), 'left')

    def test_extra_fingers_cannot_change_new_forward(self):
        def with_fingers(frame):
            direction = None if frame < 60 else 'left' if frame < 90 else 'forward'
            p, q, s = sample(direction)
            return (np.concatenate((p, np.full((116, 3), np.nan))),
                    np.concatenate((q, np.full((116, 2), 99999.))),
                    np.concatenate((s, np.ones(116))))
        self.assert_sequence(self.feed(with_fingers, 120), 'left')

    def test_held_forward_has_no_lateral_recount(self):
        self.sequence()
        state = self.feed(sample('forward'), 100)
        self.assert_sequence(state, 'left')

    def test_held_side_is_not_a_second_gesture(self):
        self.feed(sample(), 60)
        state = self.feed(sample('left'), 150)
        self.assertEqual([d for d, _ in self.events], ['left'])
        self.assertEqual(state['counts']['forward'], 0)

    def test_small_elbow_jitter_cannot_create_forward(self):
        self.feed(sample(), 60)
        self.feed(sample('left'), 30)
        def jitter(frame):
            p, q, s = sample('left')
            q[8] += [2 * math.sin(frame), math.cos(frame)]
            return p, q, s
        state = self.feed(jitter, 90)
        self.assertEqual(state['counts']['forward'], 0)

    def test_ordinary_lowering_cannot_create_forward(self):
        self.feed(sample(), 60)
        self.feed(sample('left'), 30)
        def lowering(frame):
            p, q, s = sample('left')
            q[[8, 10], 1] += (frame - 90) * 3
            return p, q, s
        state = self.feed(lowering, 30)
        self.assertEqual(state['counts']['forward'], 0)

    def test_both_hands_working_is_not_new_forward(self):
        self.feed(sample(), 60)
        self.feed(sample('left'), 30)
        p, q, s = sample('forward')
        p[9] = p[10] + [.02, .02, 0]
        q[9] = q[10] + [4, 4]
        state = self.feed((p, q, s), 60)
        self.assertEqual(state['counts']['forward'], 0)

    def test_weak_depth_without_prior_stable_point_is_not_guessed(self):
        self.feed(sample(), 60)
        state = self.feed(sample('forward'), 60)
        self.assertEqual(state['counts']['forward'], 0)

    def test_brief_passage_has_no_forward_count(self):
        self.feed(sample(), 60)
        self.feed(sample('left'), 30)
        self.feed(sample('forward'), 3)
        state = self.feed(sample(), 30)
        self.assertEqual(state['counts']['forward'], 0)

    def test_high_fps_confirmation_uses_elapsed_time(self):
        self.feed(sample(), 240, 120)
        self.feed(sample('left'), 120, 120)
        state = self.feed(sample('forward'), 13, 120)
        self.assertEqual(state['counts']['forward'], 0)
        self.assert_sequence(self.feed(sample('forward'), 30, 120), 'left')

    def test_same_rule_at_15_30_60_fps(self):
        for fps in (15, 30, 60):
            with self.subTest(fps=fps):
                self.setUp(); self.assert_sequence(self.sequence(fps=fps), 'left')

    def test_track_gap_clears_new_geometry_and_pending_votes(self):
        self.feed(sample(), 60)
        self.feed(sample('left'), 30)
        self.frame += 30
        state = self.feed(sample('forward'), 30)
        self.assertEqual(state['counts']['forward'], 0)

    def test_other_track_cannot_borrow_pointing_history(self):
        self.feed(sample(), 60)
        self.feed(sample('left'), 30)
        state = self.feed(sample('forward'), 30, tid=2)
        self.assertEqual(state['counts']['forward'], 0)

    def replay_real(self, name):
        events = []
        for row in SEGMENTS[name]:
            state, _ = self.rules.update(1, row['frame'], row['time_s'], row['xyz'], row['xy'], row['scores'])
            event = self.tally.observe(1, self.rules.people[1]['phase'])
            if event:
                events.append((event['direction'], event['time_s']))
        return self.tally.decorate(state), events

    def test_real_level_forward_is_restored(self):
        _, events = self.replay_real('level_front')
        self.assertEqual(len([t for d, t in events if d == 'forward' and 43.0 < t < 43.4]), 1)

    def test_real_held_side_does_not_acquire_an_extra_forward(self):
        _, events = self.replay_real('held_side')
        self.assertFalse([t for d, t in events if d == 'forward' and 15.5 < t < 16.8])

    def test_real_first_side_endpoint_is_not_premature_forward(self):
        _, events = self.replay_real('still_entering_side')
        self.assertFalse([t for d, t in events if d == 'forward' and 76.0 < t < 77.1])


if __name__ == '__main__':
    unittest.main()
