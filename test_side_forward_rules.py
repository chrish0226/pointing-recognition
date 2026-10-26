"""Real side-view regressions and optional-head fallback, without GPU inference."""
import json
from pathlib import Path
import unittest

import numpy as np
from balanced_pointing_rules import BalancedPointing
from realtime_rtmw3d import GestureTally
from test_current_body_rules import body_pose

FIXTURE=json.loads((Path(__file__).parent/'regression_fixtures/side_front_real.json').read_text(encoding='utf8'))['segments']


class SideForwardTests(unittest.TestCase):
    def replay(self,name):
        counter=BalancedPointing();tally=GestureTally()
        for row in FIXTURE['calibration']+FIXTURE[name]:
            xy=np.zeros((133,2));scores=np.zeros(133)
            xy[:17]=row['xy'];scores[:17]=row['scores']
            xy[91:133]=row['hands_xy'];scores[91:133]=row['hands_scores']
            state,_=counter.update(1,row['frame'],row['time_s'],row['xyz'],xy,scores)
            tally.observe(1,counter.people[1]['phase'])
        return tally.decorate(state)

    def test_back_profile_forward_is_not_right(self):
        self.assertEqual(self.replay('back_profile_front')['counts'],{'forward':1,'left':0,'right':0})

    def test_right_profile_forward_is_not_left(self):
        self.assertEqual(self.replay('right_profile_front')['counts'],{'forward':1,'left':0,'right':0})

    def test_close_profile_forward_survives_shoulder_ambiguity(self):
        self.assertEqual(self.replay('close_profile_front')['counts'],{'forward':1,'left':0,'right':0})

    def test_short_frontal_arm_is_not_lateral(self):
        self.assertEqual(self.replay('short_front')['counts'],{'forward':1,'left':0,'right':0})

    def test_rising_into_a_right_point_does_not_add_forward(self):
        self.assertEqual(self.replay('right_transition')['counts'],{'forward':0,'left':0,'right':1})

    def test_lowering_forward_does_not_add_right(self):
        self.assertEqual(self.replay('lowering_front')['counts'],{'forward':1,'left':0,'right':0})

    def test_missing_head_and_hand_points_fall_back_to_body(self):
        for direction in ('left','right','forward'):
            with self.subTest(direction=direction):
                counter=BalancedPointing()
                for frame in range(50):
                    p,q,s=body_pose(direction if frame>=20 else None)
                    p[:5]=np.nan;q[:5]=np.nan;s[:5]=0
                    state,_=counter.update(1,frame,frame/30,p,q,s)
                self.assertEqual(state['done'],[direction])

    def test_pose_arrays_are_not_changed_by_geometry_repair(self):
        counter=BalancedPointing()
        for row in FIXTURE['short_front']:
            arrays=[np.array(row[k]) for k in ('xyz','xy','scores')]
            copies=[v.copy() for v in arrays]
            counter.update(1,row['frame'],row['time_s'],*arrays)
            for current,original in zip(arrays,copies):
                np.testing.assert_array_equal(current,original)

    def test_auxiliary_diagnostics_can_be_written_to_json(self):
        counter=BalancedPointing()
        for row in FIXTURE['calibration']+FIXTURE['back_profile_front']:
            state,arms=counter.update(1,row['frame'],row['time_s'],row['xyz'],row['xy'],row['scores'])
            json.dumps({'state':state,'arms':arms,'observation':counter.people[1]['last_observation']})


if __name__=='__main__':unittest.main()
