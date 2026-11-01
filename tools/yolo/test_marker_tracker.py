"""Behavioral checks for identity retention, deletion and conservative births."""
import unittest

import numpy as np
from marker_tracker import BallGeometry, MarkerTracker, TrackerConfig


def scene(objects):
    image=np.zeros((256,256),np.uint8)
    for x,y,seed in objects:
        patch=(np.random.default_rng(seed).random((20,20))>0.72).astype(np.uint8)*255
        image[y-10:y+10,x-10:x+10]=patch
    return image


def detection(x,y,score=0.95):
    return dict(xyxy_pixels=[x-12,y-12,x+12,y+12],confidence=score)


class TrackingBehavior(unittest.TestCase):
    def test_high_confidence_requires_two_observations(self):
        tracker=MarkerTracker()
        self.assertFalse(tracker.update(scene([(110,120,1)]),[detection(110,120)],0)['tracks'])
        output=tracker.update(scene([(111,120,1)]),[detection(111,120)],500)
        self.assertEqual(len(output['tracks']),1)
        self.assertEqual(output['tracks'][0]['source'],'detection_high')

    def test_low_scores_cannot_initialize_tracks(self):
        tracker=MarkerTracker()
        for i in range(8):
            self.assertFalse(tracker.update(scene([(110,120,1)]),[detection(110,120,.6)],i*500)['tracks'])
        self.assertEqual(tracker.next_id,1)

    def test_tentative_survives_verified_detector_gap_but_needs_second_high_hit(self):
        tracker=MarkerTracker()
        for i in range(5):
            ds=[detection(100+i*2,120)] if i in (0,4) else []
            result=tracker.update(scene([(100+i*2,120,1)]),ds,i*500)
            if i<4:self.assertFalse(result['tracks'])
        self.assertEqual(result['tracks'][0]['track_id'],1)
        self.assertEqual(result['tracks'][0]['detector_hits'],2)
        self.assertEqual(tracker.next_id,2)

    def test_one_high_hit_followed_only_by_texture_never_confirms(self):
        tracker=MarkerTracker()
        for i in range(12):
            result=tracker.update(scene([(100+i,120,1)]),[detection(100,120)] if i==0 else [],i*500)
            self.assertFalse(result['tracks'])
        self.assertFalse(tracker.tracks)
        self.assertIn('tentative_confirmation_timeout',[r['reason'] for r in tracker.terminated])

    def test_recovery_uses_last_real_image_without_filling_blank_frame(self):
        tracker=MarkerTracker()
        tracker.update(scene([(100,120,1)]),[detection(100,120)],0)
        tracker.update(scene([(102,120,1)]),[detection(102,120)],500)
        gap=tracker.update(scene([]),[],1000)
        self.assertEqual(gap['tracks'][0]['source'],'prediction_hold')
        result=tracker.update(scene([(106,120,1)]),[],1500)
        self.assertEqual(result['tracks'][0]['track_id'],1)
        self.assertEqual(result['tracks'][0]['source'],'optical_flow')
        self.assertEqual(result['tracks'][0]['flow_method'],'last_evidence_seeded_lk')

    def test_nearby_tentative_candidate_cannot_steal_established_detection(self):
        tracker=MarkerTracker()
        for i in range(2):
            result=tracker.update(scene([(110+i*2,120,1)]),[detection(110+i*2,120)],i*500)
        identity=result['tracks'][0]['track_id']
        tracker.update(scene([(114,120,1),(145,120,2)]),[detection(114,120),detection(145,120)],1000)
        result=tracker.update(scene([(116,120,1),(140,120,2)]),[detection(116,120)],1500)
        observed=[t for t in result['tracks'] if t['source']=='detection_high']
        self.assertEqual([t['track_id'] for t in observed],[identity])

    def test_detector_dropouts_preserve_identity_via_image_support(self):
        tracker=MarkerTracker()
        identifiers=[]; sources=[]
        for i in range(8):
            ds=[detection(100+i*2,120)] if i not in (2,3,4) else []
            result=tracker.update(scene([(100+i*2,120,1)]),ds,i*500)
            if i:
                self.assertEqual(len(result['tracks']),1)
                identifiers.append(result['tracks'][0]['track_id'])
                sources.append(result['tracks'][0]['source'])
        self.assertEqual(len(set(identifiers)),1)
        self.assertIn('optical_flow',sources)

    def test_complete_cluster_disappearance_ends_track(self):
        tracker=MarkerTracker()
        for i in range(2):
            tracker.update(scene([(110,120,1)]),[detection(110,120)],i*500)
        for i in range(2,6):
            output=tracker.update(scene([]),[],i*500)
        self.assertFalse(output['tracks'])
        self.assertIn('cluster_evidence_lost',[r['reason'] for r in tracker.terminated])

    def test_low_score_association_preserves_confirmed_identity(self):
        tracker=MarkerTracker()
        ids=[]
        for i,score in enumerate([.95,.95,.40,.38,.90]):
            result=tracker.update(scene([(110+i,120,1)]),[detection(110+i,120,score)],i*500)
            if i:
                self.assertEqual(len(result['tracks']),1)
                ids.append(result['tracks'][0]['track_id'])
        self.assertEqual(len(set(ids)),1)

    def test_ball_edge_is_not_image_border(self):
        tracker=MarkerTracker()
        for i in range(3):
            result=tracker.update(scene([(228,128,1)]),[detection(228,128)],i*500)
        self.assertFalse(result['tracks'])
        self.assertTrue(result['rejected_detections'])

    def test_timestamp_gap_resets_identity_and_deletes_old_state(self):
        tracker=MarkerTracker()
        tracker.update(scene([(110,120,1)]),[detection(110,120)],0)
        old=tracker.update(scene([(110,120,1)]),[detection(110,120)],500)['tracks'][0]['track_id']
        reset=tracker.update(scene([(110,120,1)]),[detection(110,120)],10000)
        self.assertTrue(reset['reset_due_to_gap'])
        self.assertFalse(reset['tracks'])
        new=tracker.update(scene([(110,120,1)]),[detection(110,120)],10500)['tracks'][0]['track_id']
        self.assertNotEqual(old,new)

    def test_non_monotonic_input_rejected(self):
        tracker=MarkerTracker()
        tracker.update(scene([]),[],1000)
        with self.assertRaises(ValueError):
            tracker.update(scene([]),[],1000)

    def test_two_objects_keep_ids_when_detection_order_reverses(self):
        tracker=MarkerTracker(); mapping=[]
        for i in range(8):
            objects=[(95+i,100,1),(150-i,145,2)]
            ds=[detection(95+i,100),detection(150-i,145)]
            if i%2:
                ds.reverse()
            result=tracker.update(scene(objects),ds,i*500)
            if i:
                self.assertEqual(len(result['tracks']),2)
                mapping.append([t['track_id'] for t in sorted(result['tracks'],key=lambda t:t['center_pixels'][1])])
        self.assertTrue(all(m==mapping[0] for m in mapping))

    def test_crossing_objects_retain_motion_identity(self):
        tracker=MarkerTracker()
        for i in range(16):
            a,b=90+5*i,165-5*i
            result=tracker.update(scene([(a,120,1),(b,120,2)]),
                                  [detection(a,120,.96),detection(b,120,.92)],i*500)
        by_id={t['track_id']:t for t in result['tracks']}
        self.assertEqual(set(by_id),{1,2})
        self.assertGreater(by_id[1]['center_pixels'][0],by_id[2]['center_pixels'][0])
        self.assertEqual(tracker.next_id,3)

    def test_blank_image_hallucination_cannot_keep_track_alive(self):
        tracker=MarkerTracker()
        for i in range(2):
            tracker.update(scene([(110,120,1)]),[detection(110,120)],i*500)
        for i in range(2,7):
            result=tracker.update(scene([]),[detection(110,120,.99)],i*500)
        self.assertFalse(result['tracks'])


if __name__=='__main__':
    unittest.main(verbosity=2)
