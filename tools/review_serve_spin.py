import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.serve_spin import analyze_serves
from pingpong_event_pipeline.serve_review import write_review


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='逐次发球、共同时间窗口、双视角共享角速度与失败原因审核')
    parser.add_argument('--tracks',type=Path,default=root/'output/latest/tracking/rotation_tracks_all.csv')
    parser.add_argument('--dataset',type=Path,default=root/'data/server_marker_two_camera_train_v1')
    parser.add_argument('--config',type=Path,default=root/'config/offline_marker_spin.json')
    parser.add_argument('--previous-report',type=Path,default=root/'output/latest_spin/spin/spin_report.json')
    parser.add_argument('--orientation-profile',type=Path,help='实测 camera_to_reference 正旋转矩阵 JSON；不提供则做留出发球的实验性方向配准')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-videos',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): parser.error('output already exists; use a new output directory')
    report=analyze_serves(args.tracks,args.dataset,json.loads(args.config.read_text()),json.loads(args.previous_report.read_text()) if args.previous_report else None,args.orientation_profile)
    rows=write_review(report,args.tracks,args.output,not args.no_videos)
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    print(args.output/'review.html')


if __name__=='__main__':
    main()
