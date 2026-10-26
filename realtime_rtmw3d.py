"""Live RTMW3D-L TensorRT FP16 with current body-relative gesture directions.

Examples (use .venv_rtmw3d_trt/Scripts/python.exe):
    python realtime_rtmw3d.py --source 0
    python realtime_rtmw3d.py --source "rtsp://host/stream"
    python realtime_rtmw3d.py --source "sample.mp4"

Local files are processed frame by frame. Camera/network inputs use a single
latest-frame slot, with capture timestamps rather than inference-loop time.
"""
import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import ImageFont
from balanced_pointing_rules import REQUIRED

ROOT = Path(__file__).resolve().parent
WINDOW = 'RTMW3D-L FP16 | Q/Esc: quit | R: reset'
DIRECTION_NAMES = {'forward': '前', 'left': '左', 'right': '右'}


def rule_fingerprint():
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ('balanced_pointing_rules.py', 'arm_geometry.py')}
    bundle = hashlib.sha256(json.dumps(sources, sort_keys=True).encode('utf-8')).hexdigest()
    return {'rule_sha256': sources['balanced_pointing_rules.py'],
            'rule_sources_sha256': sources, 'rule_bundle_sha256': bundle}


class GestureTally:
    """Count confirmed phases, including repeated directions, never held frames.

    Observe the rule's existing phase confirmations without changing its
    thresholds, segmentation, direction decisions or completion state.
    """
    def __init__(self):
        self.people = {}

    def _person(self, tid):
        return self.people.setdefault(int(tid), {
            'counts': {direction: 0 for direction in REQUIRED},
            'token': None, 'last_gesture': None})

    def observe(self, tid, phase):
        person = self._person(tid)
        if not phase:
            return None
        token = (phase['confirmed_frame'], phase['direction'], phase['side'])
        if token == person['token']:
            return None
        direction = phase['direction']
        person['counts'][direction] += 1
        person['token'] = token
        person['last_gesture'] = {'direction': direction,
            'count': person['counts'][direction], 'frame': phase['confirmed_frame'],
            'time_s': phase['confirmed_s']}
        return {**person['last_gesture'], 'counts': dict(person['counts'])}

    def decorate(self, state):
        person = self._person(state['track_id'])
        return {**state, 'counts': dict(person['counts']),
                'last_gesture': dict(person['last_gesture']) if person['last_gesture'] else None}


def count_label(state, now):
    counts = state['counts']
    label = f'ID{state["track_id"]} ' + '  '.join(
        f'{DIRECTION_NAMES[d]}:{counts[d]}' for d in REQUIRED)
    recent = state.get('last_gesture')
    if recent and 0 <= now - recent['time_s'] <= .8:
        label += f'  [{DIRECTION_NAMES[recent["direction"]]}+1]'
    return label


def parse_source(value):
    return int(value) if str(value).isdecimal() else str(value)


def is_live_source(source):
    return isinstance(source, int) or '://' in source


def source_label(source):
    """Do not put stream credentials or query tokens into our logs."""
    if isinstance(source, int):
        return f'camera:{source}'
    if '://' not in source:
        return str(Path(source).resolve())
    parts = urlsplit(source)
    return f'{parts.scheme}://{parts.netloc.rsplit("@", 1)[-1]}{parts.path}'


@dataclass
class FramePacket:
    index: int
    image: np.ndarray
    time_s: float
    captured_at: float


class LatestFrameSlot:
    """A bounded, closeable mailbox. Replaced unread frames are counted."""
    def __init__(self):
        self.condition = threading.Condition()
        self.packet = None
        self.closed = False
        self.error = None
        self.dropped = 0

    def put(self, packet):
        with self.condition:
            if self.closed:
                return
            if self.packet is not None:
                self.dropped += 1
            self.packet = packet
            self.condition.notify_all()

    def get(self, timeout):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.packet is not None or self.closed, timeout)
            if not ready:
                raise TimeoutError('No new frame received within the capture timeout')
            if self.packet is not None:
                packet, self.packet = self.packet, None
                return packet
            if self.error:
                raise RuntimeError(self.error)
            return None

    def finish(self, error=None):
        with self.condition:
            self.closed = True
            self.error = error
            self.condition.notify_all()


class FrameReader:
    def __init__(self, args):
        self.source = parse_source(args.source)
        self.live = is_live_source(self.source)
        self.slot = LatestFrameSlot()
        self.stopped = threading.Event()
        self.thread = None
        self.index = 0
        self.started_at = time.perf_counter()
        self.timeout = args.read_timeout
        if isinstance(self.source, int):
            backend = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW,
                       'msmf': cv2.CAP_MSMF}[args.camera_backend]
            self.cap = cv2.VideoCapture(self.source, backend)
            for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, args.width),
                                (cv2.CAP_PROP_FRAME_HEIGHT, args.height),
                                (cv2.CAP_PROP_FPS, args.camera_fps)):
                if value:
                    self.cap.set(prop, value)
        elif self.live:
            timeout_ms = int(self.timeout * 1000)
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
        else:
            if not Path(self.source).is_file():
                raise FileNotFoundError(f'Input video does not exist: {self.source}')
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError('Cannot open input. Check camera index/backend, file path or stream address.')
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and 0 < fps <= 240 else 30.
        if self.live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend-dependent hint.
            self.thread = threading.Thread(target=self._capture, daemon=True)
            self.thread.start()

    def _read(self):
        ok, frame = self.cap.read()
        captured = time.perf_counter()
        if not ok:
            return None
        timestamp = captured - self.started_at if self.live else self.index / self.fps
        packet = FramePacket(self.index, frame, timestamp, captured)
        self.index += 1
        return packet

    def _capture(self):
        try:
            while not self.stopped.is_set():
                packet = self._read()
                if packet is None:
                    self.slot.finish('Live capture stopped or disconnected. Check the source and restart.')
                    return
                self.slot.put(packet)
        except Exception as exc:
            self.slot.finish(f'Capture failed: {type(exc).__name__}')
        finally:
            # Only this thread owns live cap.read()/release().
            self.cap.release()

    def read(self):
        return self.slot.get(self.timeout + 1) if self.live else self._read()

    def close(self):
        self.stopped.set()
        self.slot.finish()
        if self.thread:
            self.thread.join(timeout=min(self.timeout + 1, 2.))
        else:
            self.cap.release()


class LivePipeline:
    """Same detector, crop/decode, ByteTrack and rules as the offline batch."""
    def __init__(self, fps, detector=None, pose=None):
        from rtmw3d_trt_pointing import DetectorTRT, PoseTRT, make_tracker
        from balanced_pointing_rules import BalancedPointing
        self.detector = detector if detector is not None else DetectorTRT()
        self.pose = pose if pose is not None else PoseTRT()
        self.fps = fps
        self.tracker = make_tracker(fps)
        self.counter = BalancedPointing()
        self.tally = GestureTally()
        self.generation = 0

    def reset(self):
        from rtmw3d_trt_pointing import make_tracker
        from balanced_pointing_rules import BalancedPointing
        self.counter.close()
        self.tracker = make_tracker(self.fps)
        self.counter = BalancedPointing()
        self.tally = GestureTally()
        self.generation += 1

    def infer(self, packet):
        from ultralytics.engine.results import Boxes
        start = time.perf_counter()
        image = packet.image
        detections = self.detector(image)
        detector_ms = (time.perf_counter() - start) * 1000
        t = time.perf_counter()
        boxes = np.column_stack((detections, np.zeros(len(detections), np.float32)))
        tracks = self.tracker.update(Boxes(boxes, image.shape[:2]), image)
        tracking_ms = (time.perf_counter() - t) * 1000
        people, changes = [], []
        pose_ms, rule_ms = 0., 0.
        for track in tracks:
            tid = int(track[4])
            previous = set(self.counter.snapshot(tid)['done']) if tid in self.counter.people else set()
            t = time.perf_counter()
            xy, scores, depth, xyz, _ = self.pose(image, track[:4])
            pose_ms += (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            # Finger/hand landmarks never enter the recognition rule boundary.
            state, arms = self.counter.update(tid, packet.index, packet.time_s,
                                             xyz[:17], xy[:17], scores[:17])
            confirmation = self.tally.observe(tid, self.counter.people[tid]['phase'])
            state = self.tally.decorate(state)
            rule_ms += (time.perf_counter() - t) * 1000
            reference = self.counter.people[tid]['reference']
            observation = self.counter.people[tid]['last_observation'] or {}
            people.append({'track_id': tid, 'bbox': track[:4].round(2).tolist(),
                'detection_confidence': float(track[5]),
                'keypoints_xy': xy[:17].round(3).tolist(),
                'keypoint_confidence': scores[:17].round(5).tolist(),
                'keypoints_3d_assumed_camera': xyz[:17].round(6).tolist(),
                'directions': [a['direction'] for a in arms if a['selected']],
                'arms': arms,
                'body_reference': {key: value.tolist() if isinstance(value, np.ndarray) else value
                                   for key, value in reference.items()} if reference else None,
                'motion': {key: observation.get(key) for key in
                           ('walking', 'hip_speed', 'foot_speed', 'reference_settling')},
                'orientation': {key: observation.get(key) for key in
                                ('orientation_assist_ready', 'orientation_cosine', 'orientation_reference')},
                'state': state})
            if confirmation:
                changes.append({'type': 'direction', 'frame': packet.index,
                    'time_s': packet.time_s, 'track_id': tid,
                    'direction': confirmation['direction'],
                    'count': confirmation['count'], 'counts': confirmation['counts'],
                    'new_direction': confirmation['direction'] not in previous,
                    'state': state})
        return people, changes, {'detector_ms': detector_ms, 'pose_ms': pose_ms,
            'tracking_ms': tracking_ms, 'rules_ms': rule_ms,
            'inference_ms': (time.perf_counter() - start) * 1000}

    def retire(self, now, ttl):
        """Remove long-lost identities to bound memory in indefinite sessions."""
        expired = []
        for tid, state in list(self.counter.people.items()):
            if now - state['last_seen_s'] > ttl:
                expired.append({'type': 'track_expired', 'time_s': now,
                                'state': self.tally.decorate(self.counter.snapshot(tid))})
                del self.counter.people[tid]
                self.tally.people.pop(tid, None)
        return expired

    def summary(self):
        return [self.tally.decorate(person) for person in self.counter.summary()]


def load_font(path=None):
    candidates = [Path(path)] if path else [Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), 20)
    raise FileNotFoundError('Chinese font unavailable; specify a .ttf/.ttc file with --font')


def fit_display(frame, max_width):
    if max_width and frame.shape[1] > max_width:
        return cv2.resize(frame, (max_width, round(frame.shape[0] * max_width / frame.shape[1])))
    return frame


def window_is_open():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='0', help='Camera index, RTSP/HTTP URL, or local video file')
    ap.add_argument('--camera-backend', choices=('auto', 'dshow', 'msmf'), default='auto')
    ap.add_argument('--width', type=int, default=0, help='Requested camera width; 0 keeps driver default')
    ap.add_argument('--height', type=int, default=0)
    ap.add_argument('--camera-fps', type=float, default=0)
    ap.add_argument('--read-timeout', type=float, default=8., help='Network open/read timeout in seconds')
    ap.add_argument('--no-display', action='store_true', help='Run without a GUI window')
    ap.add_argument('--display-width', type=int, default=1280, help='Resize display only; 0 = original')
    ap.add_argument('--font', help='Chinese font path')
    ap.add_argument('--save', type=Path, help='Optional .mp4 recording (mp4v, processed frames only)')
    ap.add_argument('--record-fps', type=float, default=0, help='Recording FPS; 0 = nominal input FPS')
    ap.add_argument('--log', type=Path, help='Optional JSONL event log; summary saved alongside it')
    ap.add_argument('--log-frames', action='store_true', help='Also log body points/states for each processed frame')
    ap.add_argument('--max-frames', type=int, default=0, help='Stop after N processed frames; 0 = unlimited')
    ap.add_argument('--state-ttl', type=float, default=60., help='Forget a lost track after this many seconds')
    ap.add_argument('--file-realtime', action='store_true', help='Pace file playback at input FPS, without skipping')
    return ap


def run(args):
    # Import TensorRT only after parsing --help. Torch loads CUDA DLLs first.
    from rtmw3d_trt_pointing import DetectorTRT, PoseTRT
    from rtmw_trt_runtime import torch, TRT_DIR, trt
    from yolo_pointing_simple import render
    from balanced_pointing_rules import DEFAULTS, PROFILE
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Use the project .venv_rtmw3d_trt interpreter.')
    torch.set_num_threads(4)
    font = load_font(args.font)
    for engine in ('rtmw3d_l_fp16.engine', 'yolox_s_fp16.engine'):
        if not (TRT_DIR / engine).is_file():
            raise FileNotFoundError(f'Missing engine: {TRT_DIR / engine}')
    output_paths = [path.resolve() for path in (args.save, args.log,
                    args.log.with_suffix('.summary.json') if args.log else None) if path]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError('Recording, event log and summary paths must differ')
    if not is_live_source(parse_source(args.source)) and Path(args.source).resolve() in output_paths:
        raise ValueError('Output paths must differ from the input video')
    if args.save and args.save.suffix.lower() != '.mp4':
        raise ValueError('--save must end in .mp4')
    if args.log_frames and not args.log:
        raise ValueError('--log-frames requires --log')
    print('Loading RTMW3D-L / YOLOX-s TensorRT FP16 engines...', flush=True)
    detector, pose = DetectorTRT(), PoseTRT()
    reader, pipeline, writer, log = None, None, None, None
    processed = 0
    status = 'running'
    session_wall = None
    inference_times = deque(maxlen=60)
    loop_times = deque(maxlen=60)
    next_print = 0.
    last_size = None
    reset_count = 0

    def emit(event):
        if log:
            log.write(json.dumps({'generation': pipeline.generation if pipeline else 0,
                                  **event}, ensure_ascii=False, separators=(',', ':')) + '\n')
            if event['type'] != 'frame':
                log.flush()

    try:
        reader = FrameReader(args)
        pipeline = LivePipeline(reader.fps, detector, pose)
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open('w', encoding='utf8')
        emit({'type': 'session_start', 'source': source_label(reader.source),
              'model': 'RTMW3D-L TensorRT FP16', 'rules': DEFAULTS, 'rule_profile': PROFILE,
              **rule_fingerprint(),
              'source_fps': reader.fps, 'live_latest_frame': reader.live})
        packet = reader.read()
        if packet is None:
            raise RuntimeError('Input has no readable frame')
        print('Warming up engines...', flush=True)
        for _ in range(3):
            boxes = detector(packet.image)
            h, w = packet.image.shape[:2]
            pose(packet.image, boxes[0, :4] if len(boxes) else np.array([0, 0, w-1, h-1]))
        if reader.live:
            packet = reader.read()  # Discard stale warmup input for live capture.
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        session_wall = time.perf_counter()
        pacing_start = session_wall
        print(f'Source: {source_label(reader.source)} | nominal FPS {reader.fps:.2f}', flush=True)
        print('Red until front+left+right; green stays latched. Q/Esc quit; R resets all people.', flush=True)
        while packet is not None:
            loop_start = time.perf_counter()
            size = (packet.image.shape[1], packet.image.shape[0])
            if last_size is not None and size != last_size:
                if writer:
                    raise RuntimeError('Input resolution changed during recording; restart with a new recording file')
                emit({'type': 'reset', 'reason': 'resolution_change', 'time_s': packet.time_s})
                pipeline.reset()
                reset_count += 1
            last_size = size
            people, changes, timings = pipeline.infer(packet)
            for change in changes:
                emit(change)
                counts = change['counts']
                print(f't={packet.time_s:.2f}s ID{change["track_id"]}: '
                      f'{change["direction"]} +1 (#{change["count"]}) | '
                      f'F={counts["forward"]} L={counts["left"]} R={counts["right"]} | '
                      f'{"GREEN" if change["state"]["passed"] else "RED"}', flush=True)
            for event in pipeline.retire(packet.time_s, args.state_ttl):
                emit(event)
            canvas = render(packet.image, people, font,
                            label_fn=lambda state: count_label(state, packet.time_s))
            inference_times.append(timings['inference_ms'])
            age_ms = max(0., (time.perf_counter() - packet.captured_at) * 1000)
            infer_fps = 1000 / max(np.mean(inference_times), .001)
            hud = f'Infer {infer_fps:.1f} FPS | {timings["inference_ms"]:.0f} ms | People {len(people)}'
            if reader.live:
                hud += f' | Frame age {age_ms:.0f} ms | Skip {reader.slot.dropped}'
            cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1100), 30), (20, 20, 20), -1)
            cv2.putText(canvas, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            if args.save and writer is None:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*'mp4v'),
                                         args.record_fps or reader.fps, size)
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot create recording: {args.save}')
            if writer:
                writer.write(canvas)
            if args.log_frames:
                emit({'type': 'frame', 'frame': packet.index, 'time_s': packet.time_s,
                      'people': people, 'timings': timings, 'frame_age_ms': age_ms if reader.live else None})
            key = -1
            if not args.no_display:
                cv2.imshow(WINDOW, fit_display(canvas, args.display_width))
                key = cv2.waitKey(1) & 0xff
            processed += 1
            loop_times.append((time.perf_counter() - loop_start) * 1000)
            now = time.perf_counter()
            if now >= next_print:
                print(f'frames={processed} pipeline={1000 / max(np.mean(loop_times), .001):.1f} FPS '
                      f'infer={infer_fps:.1f} FPS people={len(people)}', flush=True)
                next_print = now + 2
            if key in (ord('q'), ord('Q'), 27):
                status = 'user_exit'
                break
            if not args.no_display and not window_is_open():
                status = 'window_closed'
                break
            if args.max_frames and processed >= args.max_frames:
                status = 'max_frames'
                break
            if key in (ord('r'), ord('R')):
                emit({'type': 'reset', 'reason': 'keyboard', 'time_s': packet.time_s})
                pipeline.reset()
                reset_count += 1
                print('All direction states and track IDs reset.', flush=True)
            if not reader.live and args.file_realtime:
                due = pacing_start + (packet.index + 1) / reader.fps
                remaining = due - time.perf_counter()
                if remaining > 0:
                    time.sleep(min(remaining, 1 / reader.fps))
            packet = reader.read()
        if status == 'running':
            status = 'end_of_file'
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception:
        status = 'error'
        emit({'type': 'error', 'message': 'Inference stopped; see console traceback.'})
        raise
    finally:
        elapsed = time.perf_counter() - session_wall if session_wall is not None else 0.
        if reader:
            reader.close()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if pipeline:
            pipeline.counter.close()
            summary = {'source': source_label(reader.source) if reader else args.source,
                'status': status, 'processed_frames': processed, 'elapsed_s': elapsed,
                'wall_processing_fps': processed / elapsed if elapsed else 0,
                'live_frames_replaced': reader.slot.dropped if reader else 0,
                'resets': reset_count, 'model': 'RTMW3D-L TensorRT FP16',
                'gpu': torch.cuda.get_device_name(0), 'tensorrt': trt.__version__,
                'rules': pipeline.counter.config,
                'rule_profile': PROFILE,
                **rule_fingerprint(),
                'people_current_generation': pipeline.summary(),
                'counting': 'One increment per confirmed gesture phase, including repeats and gestures after green; reset with R.',
                'notes': ['Lost tracks expire after state-ttl; expiry and direction events are in JSONL when enabled.',
                          'infer FPS excludes drawing, capture, display and recording; wall FPS includes waits.',
                          'Frame age starts after capture/decode, not at camera exposure.',
                          'Monocular XYZ remains uncalibrated; no separate face model or manual ROI.',
                          'Existing nose/ear points optionally assist heading; missing points fall back to body rules.',
                          'Gesture evidence uses shoulder/elbow/wrist only; hand/finger landmarks are excluded at the rule input boundary.',
                          'Continuous side-to-front changes use projected shortening, height consistency and depth evidence, without a required new raise.']}
            emit({'type': 'session_end', 'status': status, 'processed_frames': processed})
            if args.log:
                summary_path = args.log.with_suffix('.summary.json')
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
                print(f'Summary: {summary_path.resolve()}', flush=True)
        if log:
            log.close()
        print(f'Stopped: {status}, processed {processed} frames.', flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.width, args.height, args.camera_fps, args.record_fps, args.max_frames, args.display_width) < 0:
        parser.error('Size, FPS and frame limits must be non-negative')
    if args.read_timeout <= 0 or args.state_ttl <= 0:
        parser.error('Timeout and state TTL must be positive')
    run(args)


if __name__ == '__main__':
    main()
