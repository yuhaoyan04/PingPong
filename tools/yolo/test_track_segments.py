"""Tests for the scientific-export boundary: count observations, never fabricated holds."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from track_segments import select_consecutive_segments,write_rotation_exports


def sample(index,source='detection_high',track_id=1,timestamp=None,camera='F3E',group='serve02_F3E'):
    return dict(camera=camera,group=group,image=f'{index:05d}.png',frame_index=index,
                timestamp_us=index*500 if timestamp is None else timestamp,tracks=[dict(
                    track_id=track_id,source=source,xyxy_pixels=[100,100,120,120],center_pixels=[110,110],
                    detection_confirmed_this_frame=source.startswith('detection'),last_detection_confidence=.9,
                    trail=[[90,90],[110,110]])])


class ScientificSegmentFilter(unittest.TestCase):
    def select(self,frames,minimum=3):
        return select_consecutive_segments(frames,minimum,750)

    def test_one_and_two_samples_removed(self):
        for n in (1,2):
            frames,segments,rejected,summary=self.select([sample(i) for i in range(n)])
            self.assertFalse(segments)
            self.assertFalse(any(f['tracks'] for f in frames))
            self.assertEqual(summary['discarded_short_samples'],n)
            self.assertTrue(all(r['confirmation_timestamp_us'] is None for r in rejected))

    def test_exactly_three_keeps_all_points_including_initial_two(self):
        frames,segments,_,summary=self.select([sample(i) for i in range(3)])
        self.assertEqual(summary['retained_samples'],3)
        self.assertEqual([f['timestamp_us'] for f in frames],[0,500,1000])
        self.assertEqual(segments[0]['confirmation_timestamp_us'],1000)
        self.assertEqual([f['tracks'][0]['segment_frame_offset'] for f in frames],[0,1,2])

    def test_longer_run_retains_every_point(self):
        frames,segments,_,_=self.select([sample(i) for i in range(7)])
        self.assertEqual(segments[0]['segment_length'],7)
        self.assertTrue(all(len(f['tracks'])==1 for f in frames))

    def test_optical_flow_with_image_evidence_counts(self):
        _,segments,_,_=self.select([sample(0),sample(1,'optical_flow'),sample(2,'optical_flow')])
        self.assertEqual(segments[0]['sources']['optical_flow'],2)

    def test_prediction_padding_cannot_qualify_short_detection(self):
        frames,segments,_,summary=self.select([sample(0),sample(1,'prediction_hold'),sample(2,'prediction_hold')])
        self.assertFalse(segments)
        self.assertEqual(summary['prediction_only_samples_excluded'],2)

    def test_prediction_gap_splits_observations_even_same_id(self):
        frames=[sample(i,'prediction_hold' if i==2 else 'detection_high') for i in range(5)]
        _,segments,rejected,_=self.select(frames)
        self.assertFalse(segments)
        self.assertEqual([r['segment_length'] for r in rejected],[2,2])

    def test_existing_long_track_does_not_rescue_isolated_tail(self):
        frames=[sample(i,'prediction_hold' if i==3 else 'detection_high') for i in range(5)]
        selected,segments,_,summary=self.select(frames)
        self.assertEqual(len(segments),1)
        self.assertFalse(selected[-1]['tracks'])
        self.assertEqual(summary['retained_samples'],3)

    def test_absent_frame_splits_same_id(self):
        frames=[sample(i) for i in range(5)];frames[2]['tracks']=[]
        _,segments,rejected,_=self.select(frames)
        self.assertFalse(segments)
        self.assertEqual([r['segment_length'] for r in rejected],[2,2])

    def test_time_gap_breaks_run(self):
        _,segments,_,_=self.select([sample(0),sample(1),sample(2,timestamp=2000)])
        self.assertFalse(segments)

    def test_skipped_source_frame_index_breaks_run(self):
        _,segments,_,_=self.select([sample(0),sample(2,timestamp=500),sample(3,timestamp=1000)])
        self.assertFalse(segments)

    def test_camera_and_group_identity_not_joined(self):
        frames=[sample(0),sample(1),sample(2,camera='F59',group='serve02_F59')]
        self.assertFalse(self.select(frames)[1])

    def test_split_valid_runs_get_unique_segment_ids_and_separate_trails(self):
        frames=[sample(i,'prediction_hold' if i==3 else 'detection_high') for i in range(7)]
        output,segments,_,_=self.select(frames)
        self.assertEqual(len(segments),2)
        self.assertNotEqual(segments[0]['segment_id'],segments[1]['segment_id'])
        self.assertEqual(len(output[4]['tracks'][0]['trail']),1)

    def test_raw_input_not_mutated(self):
        frames=[sample(i) for i in range(4)];before=deepcopy(frames)
        self.select(frames)
        self.assertEqual(frames,before)

    def test_duplicate_identity_is_rejected(self):
        frames=[sample(0)];frames[0]['tracks']*=2
        with self.assertRaises(ValueError):self.select(frames)

    def test_stricter_minimum_allowed_but_not_less_than_three(self):
        with self.assertRaises(ValueError):self.select([sample(0)],2)
        self.assertFalse(self.select([sample(i) for i in range(4)],5)[1])

    def test_empty_export_has_csv_header(self):
        frames,segments,rejected,summary=self.select([sample(0)])
        with tempfile.TemporaryDirectory() as directory:
            write_rotation_exports(Path(directory),frames,segments,rejected,summary)
            text=(Path(directory)/'rotation_tracks.csv').read_text(encoding='utf-8-sig')
            self.assertEqual(len(text.splitlines()),1)
            self.assertIn('segment_id',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
