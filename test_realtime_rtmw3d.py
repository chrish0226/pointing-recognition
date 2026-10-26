"""Realtime input regressions. No physical camera/network/GPU is opened."""
import unittest
from unittest.mock import patch
import numpy as np
from realtime_rtmw3d import (FramePacket, LatestFrameSlot, FrameReader,
                            LivePipeline, build_parser, source_label, fit_display,
                            window_is_open, GestureTally, count_label)


class FakeCapture:
    def __init__(self):
        self.frames = iter([np.full((24,32,3), 1, np.uint8),
                            np.full((24,32,3), 2, np.uint8)])
        self.released = False

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def isOpened(self):
        return True

    def get(self, _):
        return 30.

    def set(self, *args):
        return True

    def release(self):
        self.released = True


class RealtimeInputTests(unittest.TestCase):
    def packet(self, index):
        return FramePacket(index, np.zeros((2,2,3), np.uint8), index/30, 10+index/30)

    def test_slow_consumer_receives_latest_instead_of_a_backlog(self):
        slot = LatestFrameSlot()
        for i in range(200):
            slot.put(self.packet(i))
        self.assertEqual(slot.get(.01).index, 199)
        self.assertEqual(slot.dropped, 199)
        self.assertIsNone(slot.packet)

    def test_live_disconnect_is_reported_after_the_last_pending_frame(self):
        slot = LatestFrameSlot()
        slot.put(self.packet(4))
        slot.finish('disconnected')
        self.assertEqual(slot.get(.01).index, 4)
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            slot.get(.01)

    def test_no_frame_timeout_does_not_repeat_the_last_frame(self):
        slot = LatestFrameSlot()
        slot.put(self.packet(1))
        slot.get(.01)
        with self.assertRaises(TimeoutError):
            slot.get(.001)

    def test_local_files_preserve_every_frame_and_source_timestamps(self):
        fake = FakeCapture()
        args = build_parser().parse_args(['--source', 'example.mp4'])
        with patch('realtime_rtmw3d.cv2.VideoCapture', return_value=fake), \
             patch('realtime_rtmw3d.Path.is_file', return_value=True):
            reader = FrameReader(args)
            self.assertFalse(reader.live)
            self.assertEqual(reader.read().time_s, 0.)
            second = reader.read()
            self.assertEqual(second.index, 1)
            self.assertAlmostEqual(second.time_s, 1/30)
            self.assertEqual(int(second.image[0,0,0]), 2)
            self.assertIsNone(reader.read())
            reader.close()
        self.assertTrue(fake.released)

    def test_live_reader_is_bounded_and_detects_failure(self):
        fake = FakeCapture()
        args = build_parser().parse_args(['--source', '0'])
        with patch('realtime_rtmw3d.cv2.VideoCapture', return_value=fake):
            reader = FrameReader(args)
            reader.thread.join(1)
            self.assertFalse(reader.thread.is_alive())
            latest = reader.read()
            self.assertEqual(latest.index, 1)
            self.assertGreaterEqual(latest.time_s, 0)
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                reader.read()
            reader.close()
        self.assertTrue(fake.released)

    def test_display_resize_does_not_change_inference_coordinates(self):
        original = np.zeros((720,1280,3), np.uint8)
        display = fit_display(original, 640)
        self.assertEqual(display.shape[:2], (360,640))
        self.assertEqual(original.shape[:2], (720,1280))

    def test_stream_password_and_query_tokens_are_removed_from_our_logs(self):
        label = source_label('rtsp://admin:secret@192.168.1.2:554/live?token=hidden')
        self.assertEqual(label, 'rtsp://192.168.1.2:554/live')

    def test_window_close_does_not_turn_a_normal_exit_into_an_exception(self):
        import cv2
        with patch('realtime_rtmw3d.cv2.getWindowProperty', side_effect=cv2.error('closed')):
            self.assertFalse(window_is_open())

    def test_reset_starts_a_new_round_without_retaining_direction_flags(self):
        from test_balanced_pointing_rules import pose
        current = [pose('left')]
        def fake_pose(image, box):
            xyz, xy, scores = current[0]
            return xy, scores, np.zeros(17), xyz, None
        pipeline = LivePipeline(30,
            detector=lambda image: np.array([[20,20,100,180,.9]],np.float32), pose=fake_pose)
        for i in range(25):
            people, _, _ = pipeline.infer(self.packet(i))
        self.assertEqual(people[0]['state']['done'], ['left'])
        self.assertEqual(people[0]['state']['counts'], {'forward': 0, 'left': 1, 'right': 0})
        pipeline.reset()
        current[0] = pose()
        people, _, _ = pipeline.infer(self.packet(25))
        self.assertEqual(people[0]['state']['done'], [])
        self.assertFalse(people[0]['state']['passed'])
        self.assertEqual(people[0]['state']['counts'], {'forward': 0, 'left': 0, 'right': 0})
        self.assertEqual(pipeline.generation, 1)

    def test_rule_boundary_excludes_all_hand_and_finger_landmarks(self):
        from test_natural_pointing import compact_pose
        xyz, xy, scores = compact_pose('left')
        pipeline = LivePipeline(30,
            detector=lambda image: np.array([[20,20,100,180,.9]],np.float32),
            pose=lambda image, box: (xy, scores, np.zeros(133), xyz, None))
        with patch.object(pipeline.counter, 'update', wraps=pipeline.counter.update) as update:
            people, _, _ = pipeline.infer(self.packet(0))
            self.assertEqual([a.shape for a in update.call_args.args[3:]],
                             [(17,3), (17,2), (17,)])
        self.assertNotIn('hand_keypoints_xy', people[0])
        self.assertNotIn('hand_keypoint_confidence', people[0])


class RealtimeCountTests(unittest.TestCase):
    def setUp(self):
        from test_balanced_pointing_rules import pose
        self.make_pose = pose
        self.current = pose()
        self.frame = 0
        self.events = []
        def fake_pose(image, box):
            xyz, xy, scores = self.current
            return xy, scores, np.zeros(17), xyz, None
        self.pipeline = LivePipeline(30,
            detector=lambda image: np.array([[20,20,250,290,.9]],np.float32), pose=fake_pose)
        self.feed(None, 12)

    def feed(self, direction, n, invisible=False):
        self.current = self.make_pose(direction)
        if invisible:
            self.current[2][[9,10]] = .1
        for _ in range(n):
            packet = FramePacket(self.frame, np.zeros((300,300,3),np.uint8), self.frame/30, 10+self.frame/30)
            people, events, _ = self.pipeline.infer(packet)
            self.events.extend(events)
            self.frame += 1
        return people[0]['state']

    def test_held_pose_counts_once_and_a_new_raise_counts_again(self):
        state = self.feed('left', 60)
        self.assertEqual(state['counts']['left'], 1)
        self.feed(None, 12)
        state = self.feed('left', 20)
        self.assertEqual(state['counts'], {'forward': 0, 'left': 2, 'right': 0})
        self.assertEqual(len(self.events), 2)
        self.assertFalse(self.events[-1]['new_direction'])
        self.assertEqual(self.events[-1]['count'], 2)
        self.assertFalse(state['passed'])

    def test_counts_continue_after_green(self):
        for direction in ('forward','left','right'):
            self.feed(direction, 20)
            self.feed(None, 12)
        state = self.feed('right', 20)
        self.assertTrue(state['passed'])
        self.assertEqual(state['counts'], {'forward': 1, 'left': 1, 'right': 2})
        self.assertEqual(len(self.events), 4)
        self.assertEqual(self.pipeline.summary()[0]['counts'], state['counts'])

    def test_brief_wrist_dropout_does_not_increment_a_held_gesture(self):
        self.feed('left', 20)
        self.feed('left', 3, invisible=True)
        state = self.feed('left', 20)
        self.assertEqual(state['counts']['left'], 1)

    def test_separate_people_do_not_share_counts(self):
        tally = GestureTally()
        phase = {'confirmed_frame': 12, 'confirmed_s': .4, 'direction': 'right', 'side': 'R'}
        tally.observe(1, phase)
        tally.observe(1, phase)  # The same confirmed phase is not a second action.
        self.assertEqual(tally.decorate({'track_id': 1})['counts']['right'], 1)
        self.assertEqual(tally.decorate({'track_id': 2})['counts']['right'], 0)

    def test_expired_track_reports_then_removes_its_counts(self):
        state = self.feed('left', 20)
        expired = self.pipeline.retire(100, 60)
        self.assertEqual(expired[0]['state']['counts']['left'], 1)
        self.assertNotIn(state['track_id'], self.pipeline.tally.people)

    def test_label_shows_actual_count_and_temporary_increment(self):
        state = {'track_id': 3, 'counts': {'forward': 2, 'left': 1, 'right': 3},
                 'last_gesture': {'direction': 'right', 'time_s': 5.}}
        self.assertEqual(count_label(state, 5.1), 'ID3 前:2  左:1  右:3  [右+1]')
        self.assertNotIn('+1', count_label(state, 6.))


if __name__ == '__main__':
    unittest.main()
