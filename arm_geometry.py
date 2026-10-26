"""Shoulder/elbow/wrist evidence, with no hand or finger landmarks.

Direction hypotheses compete using arm geometry and recent motion. The body
reference is provided by the caller. No scene labels or expected order enter
these functions.
"""
import numpy as np


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros_like(v)


def direction_evidence(wrist3, fore3, wrist2, fore2, angle, ref, side):
    """Return comparable scores; projection alone cannot override spatial data."""
    w = _unit(np.array([wrist3[0], wrist3[2]]))
    f = _unit(np.array([fore3[0], fore3[2]]))
    spatial = .70 * w + .30 * f
    front = max(0., float(spatial[1]))
    left = max(0., -float(spatial[0]))
    right = max(0., float(spatial[0]))
    horizontal = abs(fore2[0]) / max(float(np.linalg.norm(fore2)), 1e-8)
    # A folded arm can leave the wrist close to its shoulder while the
    # forearm points clearly across the body. Use the elbow-to-wrist ray.
    frontal = abs(ref['right'][2]) < .45 and ref['width'] > .40
    bent = angle < 110.
    own_sign = 1. if side == 'R' else -1.
    elbow_forward = wrist3[2] - fore3[2]
    low = wrist2[1] < -.25
    cross_body = ((bent if low else angle < 70.) and fore2[0] * own_sign < 0 and
                  abs(fore2[0]) > .14 and horizontal > .80 and elbow_forward < .22)
    low_outward = (fore2[0] * own_sign > 0 and wrist2[1] < -.25 and
                   wrist2[1] - fore2[1] < -.25 and abs(wrist2[0]) > .24 and
                   abs(fore2[0]) > .06 and horizontal > .60)
    spatial_side = max(left, right) > front + .15
    visible_side = abs(fore2[0]) > .10 and horizontal > .70 and spatial_side
    if frontal and (cross_body or low_outward or visible_side):
        projected = min(1., abs(fore2[0]) / (.12 if low_outward else .24)) * horizontal
        weight = .82 if cross_body or low_outward else .25
        left = (1 - weight) * left + weight * (projected if fore2[0] < 0 else 0.)
        right = (1 - weight) * right + weight * (projected if fore2[0] > 0 else 0.)
        front *= 1 - weight
    scores = {'forward': front, 'left': left, 'right': right}
    ranked = sorted(scores, key=scores.get, reverse=True)
    winner = ranked[0]
    margin = scores[winner] - scores[ranked[1]]
    supported = (wrist3[2] > .18 if winner == 'forward'
                 else abs(wrist3[0]) > .12 or abs(fore2[0]) > .10)
    direction = winner if supported and scores[winner] >= .43 and margin >= .12 else None
    return direction, {k: float(v) for k, v in scores.items()}, float(margin), bool(frontal and cross_body)


def apply_arm_motion(arms, p, q, reference, phase, histories, now):
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
    both_arms_working = (np.linalg.norm(q[9] - q[10]) < .90 * torso and
                         all((q[w] - q[sh]) @ up2 / torso > -.72
                             for sh, w in ((5, 9), (6, 10))))
    for arm in arms:
        side = arm['side']
        sh, el, wr = (5, 7, 9) if side == 'L' else (6, 8, 10)
        wrist = (q[wr] - q[sh]) / torso
        fore = (q[wr] - q[el]) / torso
        lateral = float(wrist @ right2)
        fore_extent = float(np.linalg.norm(fore))
        history = histories[side]
        while history and now - history[0]['time'] > .8:
            history.popleft()
        if both_arms_working and arm.get('forearm_side_evidence'):
            arm['direction'] = None
            arm['bimanual_suppressed'] = True
        # A side-to-front rotation can shorten the projected forearm without
        # lowering it. Record this as independent evidence for a new pose.
        if (arm['raised'] and arm['lift'] > -.64 and phase and
                phase['side'] == side and phase['direction'] in ('left', 'right') and
                not both_arms_working and abs(reference['right'][2]) < .45):
            anchor = [v for v in history if v['phase'] == phase['direction']
                      and v['time'] >= phase['confirmed_s'] and
                      abs(v['wrist_lateral']) > .30]
            if anchor:
                peak = max(anchor, key=lambda v: v['fore_extent'])
                depth_forward = float((p[wr] - p[sh]) @ reference['forward'] / torso3)
                shortening = fore_extent < .62 * peak['fore_extent']
                inward = abs(lateral) < abs(peak['wrist_lateral']) - .13
                level = arm['lift'] >= peak['lift'] - .08
                if shortening and inward and level and depth_forward > .24:
                    arm['direction'] = 'forward'
                    arm['natural_transition'] = True
        history.append({'time': now, 'wrist_lateral': lateral,
                        'fore_extent': fore_extent, 'lift': arm['lift'],
                        'phase': phase['direction'] if phase and phase['side'] == side else None})
