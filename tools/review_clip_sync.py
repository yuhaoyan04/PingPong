import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pingpong_event_pipeline.clip_sync import build_sync_review


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='核对双侧裁剪的绝对时间交集，并保留原双路全视野上下文')
    parser.add_argument('--dataset',type=Path,default=root/'data/server_marker_two_camera_train_v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if any((args.output/name).exists() for name in ['sync_report.json','paired','full_view']):
        parser.error('sync review already exists; use a fresh output directory')
    report=build_sync_review(args.dataset,args.output)
    print(json.dumps({k:report[k] for k in ['union_frames','shared_frames','single_crop_frames','physical_camera_synchronization_verified']},indent=2))


if __name__=='__main__':main()
