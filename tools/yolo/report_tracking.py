"""Verify frozen-data comparisons, save continuity examples and document deployment defaults."""
from collections import Counter
import hashlib
import json
import shutil

from common import ROOT, DATA, EXP, dump, table
import numpy as np
from PIL import Image, ImageDraw

OUT=ROOT/'experiments/tracking_v1'
COMPARISON=OUT/'comparison'


def main():
    result=json.loads((COMPARISON/'summary.json').read_text())
    assert result['status']=='completed' and result['total_frames']==1946 and len(result['groups'])==14
    review=OUT/'review'
    (review/'contact_sheets').mkdir(parents=True,exist_ok=True)
    (review/'continuity_examples').mkdir(exist_ok=True)
    totals={cam:Counter() for cam in ('F3E','F59')}
    frame_totals=0; gap_examples=[]; gap_total=0
    overview=Image.new('RGB',(1536,7*248),'#151515')
    csv_records=[]
    for index,group in enumerate(result['groups']):
        camera,name=group['camera'],group['group']
        directory=COMPARISON/camera/name
        records=json.loads((directory/'frames.json').read_text())
        assert len(records)==group['frames']==group['video_verified_frames']
        assert len(list((directory/'overlays').glob('*.jpg')))==len(records)
        assert all(b['unobserved_frames']<=3 for f in records for b in f['tracks'])
        assert all(len({b['track_id'] for b in f['tracks']})==len(f['tracks']) for f in records)
        geometry=json.loads((directory/'config.json').read_text())['geometry']
        assert all(np.linalg.norm(np.asarray(b['center_pixels'])-[geometry['center_x'],geometry['center_y']])<=geometry['radius']-12+1e-6
                   for f in records for b in f['tracks'])
        frame_totals+=len(records)
        totals[camera].update({k:v for k,v in group.items() if isinstance(v,int)})
        csv_records.append({k:v for k,v in group.items() if isinstance(v,(int,str))})
        shutil.copy2(directory/'contact_sheet.jpg',review/'contact_sheets'/(name+'.jpg'))
        middle=records[len(records)//2]
        image=Image.open(directory/'overlays'/middle['image'].replace('.png','.jpg')).resize((768,228),Image.Resampling.LANCZOS)
        ci=0 if camera=='F3E' else 1
        gi=index%7
        overview.paste(image,(ci*768,gi*248+20))
        ImageDraw.Draw(overview).text((ci*768+4,gi*248+3),f'{name} | fixed midpoint comparison',fill='white')
        # Find causal, same-ID detector -> non-detector gap -> detector confirmations.
        active={}; group_gaps=[]
        for i,f in enumerate(records):
            visible={t['track_id']:t for t in f['tracks']}
            for track_id in list(active):
                if track_id not in visible:
                    del active[track_id]
            for t in f['tracks']:
                track_id=t['track_id']
                if t['source'].startswith('detection'):
                    if track_id in active and active[track_id]['gap']:
                        g=active[track_id]
                        group_gaps.append(dict(camera=camera,group=name,track_id=track_id,
                                               start_frame_index=g['last_detector'],end_frame_index=i,
                                               missing_detector_frames=len(g['gap']),sources=g['gap']))
                    active[track_id]=dict(last_detector=i,gap=[])
                elif track_id in active:
                    active[track_id]['gap'].append(t['source'])
        gap_total+=len(group_gaps)
        if group_gaps:
            gap=group_gaps[0]
            start=max(0,gap['start_frame_index']-1)
            end=min(len(records),gap['end_frame_index']+2,start+7)
            sheet=Image.new('RGB',(1536,456*(end-start)),'#151515')
            for k,i in enumerate(range(start,end)):
                sheet.paste(Image.open(directory/'overlays'/records[i]['image'].replace('.png','.jpg')),(0,k*456))
            filename=f'{name}_ID{gap["track_id"]}_gap.jpg'
            sheet.save(review/'continuity_examples'/filename,quality=90)
            gap_examples.append(dict(**gap,example=filename))
        dump(directory/'reconfirmed_internal_gaps.json',group_gaps)
    assert frame_totals==1946
    overview.save(review/'overview_midpoints.jpg',quality=90)
    # Fresh inference must agree with replay for both independently trained camera models.
    regression={}
    for camera,group in [('F3E','serve05_F3E'),('F59','serve03_F59')]:
        fresh=OUT/f'fresh_{camera}'/camera/group
        actual=json.loads((fresh/'frames.json').read_text())
        cached=json.loads((COMPARISON/camera/group/'frames.json').read_text())
        assert len(actual)==len(cached)
        for a,b in zip(actual,cached):
            assert a['timestamp_us']==b['timestamp_us']
            assert [(t['track_id'],t['source']) for t in a['tracks']]==[(t['track_id'],t['source']) for t in b['tracks']]
            for ta,tb in zip(a['tracks'],b['tracks']):
                assert np.allclose(ta['xyxy_pixels'],tb['xyxy_pixels'],atol=0.05)
        regression[camera]=dict(frames=len(actual),same_ids_and_sources=True,box_tolerance_pixels=.05)
    single=json.loads(next((OUT/'single_image_check').glob('*/*/config.json')).read_text())
    assert single['temporal_tracking_enabled'] is False and single['tracker']['confirmation_hits']==1
    for log in ('behavior_tests.log','input_tests.log'):
        assert '\nOK\n' in (OUT/log).read_text()
    integrity=json.loads((EXP/'audit/source_integrity.json').read_text())
    assert {str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()}==integrity['files']
    old=json.loads((EXP/'deliverables_sha256.json').read_text())
    # Deployment code/README are intentionally updated; baseline experiment artifacts are frozen.
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in old.items() if p.startswith('experiments/two_camera_v1/'))
    dump(OUT/'verification.json',dict(status='passed',behavior_tests=11,input_tests=2,
                                     comparison_groups=14,comparison_frames=1946,videos_verified=14,
                                     fresh_model_regression=regression,single_image_entry_verified=True,
                                     same_id_internal_gaps_reconfirmed_by_detector=gap_total,
                                     source_dataset_unchanged=True,previous_experiment_artifacts_unchanged=True))
    table(review/'group_summary.csv',csv_records,sorted(set(k for r in csv_records for k in r)))
    dump(review/'continuity_examples.json',gap_examples)
    dump(OUT/'camera_totals.json',{c:dict(d) for c,d in totals.items()})
    overall=totals['F3E']+totals['F59']
    lines=['高阈值、球缘过滤与时序追踪应用报告','',
           '现有 F3E/F59 模型未重新训练。默认 scripts/predict_latest.sh 已切换到新的时序推理入口。',
           '原数据与原双相机实验保持不变，新增文件位于 experiments/tracking_v1/。',
           '', '默认行为',
           '1. 新轨迹：检测置信度 >=0.75，且连续两帧高分关联确认后才显示；不回填未来信息到前帧。',
           '2. 已确认轨迹：先关联高分检测，再用 >=0.25 的候选低分检测续接。低分框不能建立新轨迹。',
           '3. 关联采用位置/速度 Kalman 预测、全局一对一匹配、框重叠与事件簇模板相似度门控。',
           '4. 没有匹配检测时，用双向校验的稀疏 Lucas–Kanade 光流续接，至少 3 个一致特征点且有事件/外观支持。',
           '5. 连图像证据也不足时，最多保持 3 帧纯预测（当前 0.5 ms 采样约 1.5 ms）；第 4 帧结束。',
           '6. 光流可在没有模型框时继续，但默认最长 40 帧（约 20 ms）未获模型再次确认后结束，以限制漂移。',
           '7. 新发球/新相机使用独立追踪器；超过 5 ms 的时间间隔清空旧状态；重复或倒序时间戳报错。',
           '8. 为减少一闪而过和重复框，额外采用 0.5 IoU 去重，并要求候选框中至少 8 个事件像素。',
           '', '球体边缘过滤',
           '使用球体坐标而非图片四周边界。由本包 crop_radius_scale=1.2 得到裁剪坐标的名义球半径 128/1.2≈106.67 px。',
           '默认球心 (128,128)，内缩 12 px 得到约 94.67 px 安全半径；框中心须在安全圈内，且框网格至少 65% 在圈内。',
           '过滤也应用于光流框和预测框；预测不能绕过球缘规则继续显示。',
           '这使用裁剪几何，并非独立拟合/核验的球轮廓；fallback 错误裁剪仍可能影响过滤。',
           '真实贴边标记也可能被主动删除，这是本次优先减少误检的取舍。其他裁剪尺度须修改球心/半径参数。',
           '', '前后对照（检测框数量/轨迹框数量，不是精确率）',
           'camera frames 原始conf0.25 高阈值0.75 高阈值+过滤 稳定轨迹框 光流续接 纯预测保持']
    for cam in ('F3E','F59'):
        t=totals[cam]
        lines.append(f"{cam} {t['frames']:6d} {t['baseline_boxes_025']:12d} {t['high_threshold_boxes_before_edge']:11d} {t['high_threshold_boxes_after_filters']:13d} {t['stable_track_boxes']:10d} {t['optical_flow']:8d} {t['prediction_hold']:10d}")
    lines += [f"总计：原始输出 {overall['baseline_boxes_025']} 个框，最终时序输出 {overall['stable_track_boxes']} 个轨迹框；不能将数量下降解释为误检率下降。",
              f"算法内部共有 {gap_total} 段同 ID 的检测缺口由光流/预测连接后又获得检测确认；此统计不代表已知真实身份准确率。",
              '其他发球仍没有人工检测/身份真值，因此不计算 FP rate、recall、IDF1 或真实 ID switch 数。',
              '不能保证永不跟丢；完全不可见、快速变化、簇交叉、错误裁剪和高置信度误检仍可能造成错误。',
              '', '审核入口',
              'review/contact_sheets/：14 组各六帧等时间抽样，四列为旧检测 / 高阈值+球缘过滤 / 稳定轨迹 / 极性参考。',
              'comparison/<相机>/<发球>/comparison_slow.mp4：全部帧慢放；25 FPS 对当前图像采样是 80 倍慢放。',
              'review/continuity_examples/：连续帧示例，针对检测暂时消失又恢复时的同 ID 续接。',
              '绿色：模型在当前帧匹配；橙色：光流延续；黄色：没有当前图像证据的短时预测保持。',
              'tracks.csv / frames.json 区分各来源；last_detection_confidence 仅表示上次检测分数，绝不是预测框的当前置信度。',
              'track_lifetimes.json / lifecycle_events.json 保存轨迹创建、确认、删除原因；ID 只在同一相机的同一序列内有效。',
              '', '验证',
              '11 项追踪行为测试和 2 项输入序列测试均通过，包括短时漏检、簇消失、空图高分幻检、交叉目标、低分禁建、时间间隔和球缘过滤。',
              '1946 帧对照使用原冻结模型输出重放，后处理只使用当前与过去信息，没有使用未来帧。',
              '另外通过默认 shell 入口，对 F3E 49 帧、F59 69 帧重新执行实际模型推理，与重放结果逐帧 ID/来源一致，框坐标差异 <0.05 px。',
              '单图入口也已实际验证：只做高阈值和球缘过滤，不声称具有时序追踪。',
              '模型权重、原始数据、原双相机实验均复核未变。',
              '', '应用命令（数据须为与该相机训练预处理一致的 256×256 单通道事件图）',
              'bash scripts/predict_latest.sh F3E data/server_marker_two_camera_train_v1/unlabeled_inference/F3E/serve02_F3E/binary 5',
              'bash scripts/predict_latest.sh F59 data/server_marker_two_camera_train_v1/unlabeled_inference/F59/serve02_F59/binary 7',
              '可追加 --birth-conf 0.85 --edge-margin 16 --max-missing 2；也可直接修改 configs/inference_tracking.json。',
              '其他球体几何可追加 --ball-radius R --center-x X --center-y Y。',
              '有 manifest 时使用真实时间戳；没有 manifest 的 PNG 文件夹按自然文件名排序，默认按 2000 FPS 构造相对时间，可用 --input-fps 修改。',
              '每次运行创建新输出目录，避免覆盖此前人工审核结果。',
              '', '实现参考',
              '高低分关联思路参考 ByteTrack；这里是针对事件簇的自定义实现，不是官方 ByteTrack 或 BoT-SORT。',
              'https://arxiv.org/abs/2110.06864',
              'https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html',
              'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html']
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    readme=ROOT/'README.txt'
    content=readme.read_text()
    addition='\n当前默认应用入口已启用高阈值+球缘过滤+稳定追踪：\nbash scripts/predict_latest.sh F3E /同相机图像序列目录 5\n配置：configs/inference_tracking.json\n新对照与报告：experiments/tracking_v1/REPORT.txt\n连续帧续接示例：experiments/tracking_v1/review/continuity_examples/\n'
    if '当前默认应用入口已启用' not in content:
        readme.write_text(content+addition,encoding='utf-8')
    print('TRACKING VERIFIED',json.dumps(overall))
    print('Reconfirmed internal gaps',gap_total)
    print('Report',OUT/'REPORT.txt')


if __name__=='__main__':
    main()
