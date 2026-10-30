"""Fresh-inference entry point and source/camera checks without dataset changes."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from apply_model import input_sequence


class InputSequenceBehavior(unittest.TestCase):
    def test_natural_order_and_explicit_assumed_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['10.png','2.png','1.png']:
                Image.new('L',(256,256)).save(root/name)
            _,_,records,paths,timebase=input_sequence(root,'F3E',2000)
            self.assertEqual([p.name for p in paths],['1.png','2.png','10.png'])
            self.assertEqual([r['timestamp_us'] for r in records],[0,500,1000])
            self.assertIn('assumed',timebase)

    def test_mismatched_camera_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'binary').mkdir()
            Image.new('L',(256,256)).save(root/'binary/a.png')
            (root/'manifest.csv').write_text('group,image,frame_index,timestamp_us\nserve02_F59,binary/a.png,0,1000\n')
            with self.assertRaises(ValueError):
                input_sequence(root,'F3E',2000)


if __name__=='__main__':
    unittest.main(verbosity=2)
