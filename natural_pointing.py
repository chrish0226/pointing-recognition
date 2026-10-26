"""Optional hand evidence for compact gestures and continuous arm turns.

All features are body-relative and normalized by torso size. No expected
direction sequence, scene region, video identity or target count is used.
Missing hand landmarks leave the existing body rules available.
"""
import numpy as np


def index_pointer(points, scores, side, torso, right2, wrist):
    if len(points) < 133 or len(scores) < 133:
        return None
    base = 91 if side == 'L' else 112
    chain = np.asarray(points[base + 5:base + 9], float)
    confidence = np.asarray(scores[base + 5:base + 9], float)
    if not np.isfinite(chain).all() or not np.isfinite(confidence).all():
        return None
    if min(confidence) < .55:
        return None
    vector = chain[-1] - chain[0]
    length = float(np.linalg.norm(vector))
    path = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
    if (length < 3 or length > .30 * torso or
            np.linalg.norm(chain[0] - wrist) > .40 * torso or
            length / max(path, 1e-8) < .82):
        return None
    lateral = float(vector @ right2 / torso)
    return {'lateral': lateral, 'extent': length / torso,
            'horizontal_fraction': abs(lateral) / max(length / torso, 1e-8),
            'confidence': float(min(confidence))}


def apply_natural_evidence(arms, p, q, hand_points, hand_scores,
                           reference, phase, histories, now):
    if reference is None:
        return
    shoulder = q[[5, 6]].mean(0)
    hip = q[[11, 12]].mean(0)
    torso = float(np.linalg.norm(shoulder - hip))
    torso3 = float(np.linalg.norm(p[[5, 6]].mean(0) - p[[11, 12]].mean(0)))
    if torso < 8 or torso3 < 1e-5:
        return
    right2 = reference['right2']
    up2 = (shoulder - hip) / torso
    frontal = abs(reference['right'][2]) < .45 and reference['width'] > .45
    both_hands_working = (np.linalg.norm(q[9] - q[10]) < .90 * torso and
                         all((q[w] - q[sh]) @ up2 / torso > -.72
                             for sh, w in ((5, 9), (6, 10))))
    for arm in arms:
        side = arm['side']
        sh, el, wr = (5, 7, 9) if side == 'L' else (6, 8, 10)
        wrist = (q[wr] - q[sh]) / torso
        fore = (q[wr] - q[el]) / torso
        lateral = float(wrist @ right2)
        fore_lateral = float(fore @ right2)
        fore_extent = float(np.linalg.norm(fore))
        history = histories[side]
        while history and now - history[0]['time'] > .8:
            history.popleft()
        pointer = index_pointer(hand_points, hand_scores, side, torso, right2, q[wr])
        arm['index_pointer'] = pointer
        eligible = frontal and arm['raised'] and arm['lift'] > -.64
        # A bent arm can point across the chest without moving the wrist past
        # its shoulder. Require an independently aligned, visible index chain.
        compact = arm['lift'] < -.25 and arm['elbow_lift'] < -.25
        if (eligible and compact and not arm.get('orientation_assisted') and
                not both_hands_working and pointer is not None and
                pointer['horizontal_fraction'] > .72 and
                abs(pointer['lateral']) > .045 and abs(fore_lateral) > .08 and
                abs(fore_lateral) / max(fore_extent, 1e-8) > .70 and
                pointer['lateral'] * fore_lateral > 0):
            arm['direction'] = 'right' if fore_lateral > 0 else 'left'
            arm['natural_side'] = True
        # A side-to-front rotation often shortens the projected forearm while
        # the hand remains at the same height. This is different from dropping
        # the arm, and does not require returning to a fully lowered posture.
        if (eligible and phase and phase['side'] == side and
                phase['direction'] in ('left', 'right') and not both_hands_working):
            anchor = [v for v in history if v['phase'] == phase['direction']
                      and v['time'] >= phase['confirmed_s'] and
                      abs(v['wrist_lateral']) > .30]
            if anchor:
                peak = max(anchor, key=lambda v: v['fore_extent'])
                depth_forward = float((p[wr] - p[sh]) @ reference['forward'] / torso3)
                shortening = fore_extent < .62 * peak['fore_extent']
                inward = abs(lateral) < abs(peak['wrist_lateral']) - .13
                level = arm['lift'] >= peak['lift'] - .08
                side_finger = (pointer is not None and
                               pointer['horizontal_fraction'] > .72 and
                               abs(pointer['lateral']) > .045)
                if shortening and inward and level and depth_forward > .24 and not side_finger:
                    arm['direction'] = 'forward'
                    arm['natural_transition'] = True
        history.append({'time': now, 'wrist_lateral': lateral,
                        'fore_extent': fore_extent, 'lift': arm['lift'],
                        'phase': phase['direction'] if phase and phase['side'] == side else None})
