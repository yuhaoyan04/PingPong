"""Create a local HTML gallery for paired raw and tracked serve clips."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    groups = []
    for manifest_path in args.manifests:
        manifest_file = Path(manifest_path).resolve()
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        groups.append(payload)

    cards = []
    for group in groups:
        cards.append(f'<h2>{html.escape(group["spin_class"].upper())} — {len(group["clips"])} clips</h2>')
        for clip in group["clips"]:
            raw_path = Path(clip["actual_events_video"]).resolve()
            tracked_path = Path(clip["tracked_trajectory_video"]).resolve()
            raw_browser = raw_path.with_name(raw_path.stem + ".browser.mp4")
            tracked_browser = tracked_path.with_name(tracked_path.stem + ".browser.mp4")
            if raw_browser.exists():
                raw_path = raw_browser
            if tracked_browser.exists():
                tracked_path = tracked_browser
            raw = raw_path.relative_to(output.parent).as_posix()
            tracked = tracked_path.relative_to(output.parent).as_posix()
            detail_path = raw_path.parent / "compensated_detail" / "dual_ball_compensated_detail.browser.mp4"
            if not detail_path.exists():
                detail_path = raw_path.parent / "compensated_detail" / "dual_ball_compensated_detail.mp4"
            detail_html = ""
            if detail_path.exists():
                detail = detail_path.relative_to(output.parent).as_posix()
                detail_html = (f'<div><b>Compensated ball detail</b>'
                               f'<video controls muted loop preload="metadata" '
                               f'src="{html.escape(detail)}"></video></div>')
            master = clip["master"]; slave = clip["slave"]
            clip_id = html.escape(clip["clip_id"])
            cards.append(f'''<section class="card" data-id="{clip_id}">
  <h3>{clip_id}</h3>
  <p>source 330 ms · master track {master['duration_ms']:.0f} ms · slave track {slave['duration_ms']:.0f} ms</p>
  <div class="videos">
    <div><b>Actual event frames</b><video controls muted loop preload="metadata" src="{html.escape(raw)}"></video></div>
    <div><b>Trajectory overlay</b><video controls muted loop preload="metadata" src="{html.escape(tracked)}"></video></div>
    {detail_html}
  </div>
  <div class="decision">
    <label><input type="radio" name="{clip_id}" value="accept"> Accept</label>
    <label><input type="radio" name="{clip_id}" value="reject"> Reject</label>
    <label><input type="radio" name="{clip_id}" value="uncertain" checked> Uncertain</label>
    <input class="note" placeholder="notes: missing view / wrong circle / unclear marker">
  </div>
</section>''')
    body = "\n".join(cards)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>DVSLume Serve Clip Review</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#15171a;color:#eee;margin:24px}}
h1{{margin-bottom:4px}} .help{{color:#bbb;max-width:1000px}}
.card{{background:#23262b;padding:16px;margin:18px 0;border-radius:10px}}
.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}}
video{{display:block;width:100%;margin-top:7px;background:#000}}
.decision{{display:flex;gap:20px;align-items:center;margin-top:13px}}
.note{{flex:1;padding:7px;background:#111;color:#eee;border:1px solid #666}}
button{{position:sticky;top:10px;padding:10px 18px;font-size:16px;background:#1d8f55;color:white;border:0;border-radius:6px}}
@media(max-width:900px){{.videos{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Real serve clip review</h1>
<p class="help">依次查看完整双目事件和轨迹叠加。轨迹图中绿色圆表示当前帧直接检测，橙色圆表示短时运动预测，青色线表示历史路径；橙色圆应继续贴住球，且在球离开后及时消失。若包含球心补偿局部图，上排为主相机、下排为从相机，时间颜色由蓝到红表示由旧到新。视频按慢动作播放。</p>
<button onclick="exportReview()">Export review JSON</button>
{body}
<script>
function exportReview(){{
 const result={{created_at:new Date().toISOString(), clips:[]}};
 document.querySelectorAll('.card').forEach(card=>{{
  const id=card.dataset.id;
  const selected=card.querySelector('input[type=radio]:checked').value;
  const notes=card.querySelector('.note').value;
  result.clips.push({{clip_id:id,status:selected,notes:notes}});
 }});
 localStorage.setItem('dvsense_serve_review',JSON.stringify(result));
 const blob=new Blob([JSON.stringify(result,null,2)],{{type:'application/json'}});
 const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
 link.download='serve_clip_human_review.json'; link.click(); URL.revokeObjectURL(link.href);
}}
const saved=localStorage.getItem('dvsense_serve_review');
if(saved){{try{{JSON.parse(saved).clips.forEach(item=>{{const card=document.querySelector(`[data-id="${{item.clip_id}}"]`);if(card){{const radio=card.querySelector(`input[value="${{item.status}}"]`);if(radio)radio.checked=true;card.querySelector('.note').value=item.notes||'';}}}})}}catch(e){{}}}}
</script></body></html>'''
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"groups": len(groups), "clips": sum(len(g["clips"]) for g in groups),
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
