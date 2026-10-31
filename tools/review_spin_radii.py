import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pingpong_event_pipeline.radius_review import compute_variants, build_radius_media


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description="多球面像素半径、逐帧转速/方向叠加与同步对比视频")
    parser.add_argument("--tracks",type=Path,default=root/"output/latest/tracking/rotation_tracks_all.csv")
    parser.add_argument("--dataset",type=Path,default=root/"data/server_marker_two_camera_train_v1")
    parser.add_argument("--config",type=Path,default=root/"config/offline_marker_spin.json")
    parser.add_argument("--previous-report",type=Path,default=root/"output/latest_spin/spin/spin_report.json")
    parser.add_argument("--radius-factors",type=float,nargs="+",default=[1.,1.1,1.2,1.5])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--render-existing",action="store_true",help="复用输出目录中已计算的 variants.json，只生成可视化")
    parser.add_argument("--no-videos",action="store_true")
    args=parser.parse_args()
    if args.render_existing:
        if not (args.output/"variants.json").is_file(): parser.error("--render-existing requires variants.json")
    else:
        if args.output.exists(): parser.error("output exists; use a new output directory")
        compute_variants(args.tracks,args.dataset,json.loads(args.config.read_text()),json.loads(args.previous_report.read_text()),args.output,args.radius_factors)
    result=build_radius_media(args.output,args.tracks,not args.no_videos)
    print(json.dumps(result["matched_window_summary"],ensure_ascii=False,indent=2))
    print(args.output/"review.html")


if __name__=="__main__":
    main()
