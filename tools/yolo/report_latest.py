"""Verify both completed camera experiments and present the current workspace."""
import csv
import hashlib
import json
import shutil
import subprocess

from common import ROOT, DATA, EXP, dump, rows, table
from PIL import Image, ImageDraw


def main():
    audited = json.loads((EXP/'audit/dataset_audit.json').read_text())
    camera_summaries, group_summaries = [], []
    preview = EXP/'review/contact_sheets'
    preview.mkdir(parents=True,exist_ok=True)
    overview = Image.new('RGB',(1728,7*362),'#161616')
    for ci,camera in enumerate(('F3E','F59')):
        out=EXP/camera
        completion=json.loads((out/'training_completion.json').read_text())
        inferred=json.loads((out/'qualitative/summary.json').read_text())
        assert completion['status']=='completed' and completion['epochs']==150
        assert inferred['status']=='completed' and len(inferred['groups'])==7
        assert inferred['total_frames']==audited['cameras'][camera]['unlabeled_frames']
        weight=out/'train/weights/last.pt'
        assert hashlib.sha256(weight.read_bytes()).hexdigest()==completion['checkpoint_sha256']
        assert not (out/'train/weights/best.pt').exists()
        loss_rows=rows(out/'train/results.csv')
        assert len(loss_rows)==150
        assert not any(k.startswith(('val/','metrics/')) for k in loss_rows[0])
        assert not list(out.rglob('*confusion_matrix*')) and not list(out.rglob('*PR_curve*'))
        verify_frames=0
        for gi,g in enumerate(inferred['groups']):
            directory=out/'qualitative'/g['group']
            records=json.loads((directory/'predictions.json').read_text())
            frame_rows=rows(directory/'frames.csv')
            assert len(records)==len(frame_rows)==g['frames']==g['verified_video_frames']
            assert len(list((directory/'overlays').glob('*.jpg')))==len(records)
            assert [(r['image'],int(r['timestamp_us'])) for r in frame_rows]==[(r['image'],r['timestamp_us']) for r in records]
            assert sum(r['display_detections'] for r in records)==g['detections']
            assert sum(r['track_support']=='fallback_unverified' for r in records)==g['fallback']['frames']
            assert (directory/'reference_original.mp4').resolve().is_file()
            verify_frames+=len(records)
            shutil.copy2(directory/'contact_sheet.jpg',preview/(g['group']+'.jpg'))
            middle=records[len(records)//2]
            image=Image.open(directory/'overlays'/(middle['image'].replace('.png','.jpg'))).resize((864,342),Image.Resampling.LANCZOS)
            overview.paste(image,(ci*864,gi*362+20))
            ImageDraw.Draw(overview).text((ci*864+4,gi*362+3),f"{g['group']} | frames={g['frames']} fallback={g['fallback']['frames']} | midpoint",fill='white')
            group_summaries.append(dict(camera=camera,group=g['group'],frames=g['frames'],
                                        frames_with_predictions=g['frames_with_predictions'],frames_without_predictions=g['frames_without_predictions'],
                                        predicted_boxes=g['detections'],fallback_frames=g['fallback']['frames'],
                                        fallback_with_predictions=g['fallback']['frames_with_predictions'],
                                        supported_frames=g['ring_supported']['frames'],
                                        supported_with_predictions=g['ring_supported']['frames_with_predictions'],
                                        reference_indices_out_of_bounds=g['reference_indices_out_of_bounds']))
        assert verify_frames==inferred['total_frames']
        camera_summaries.append(dict(camera=camera,training_images=audited['cameras'][camera]['images'],
                                     training_boxes=audited['cameras'][camera]['boxes'],
                                     explicit_negatives=audited['cameras'][camera]['explicit_negatives'],
                                     training_fallback_frames=audited['cameras'][camera]['fallback_frames'],
                                     unlabeled_frames=inferred['total_frames'],
                                     unlabeled_fallback_frames=inferred['fallback_frames'],
                                     frames_with_predictions=inferred['frames_with_predictions'],
                                     frames_without_predictions=inferred['frames_without_predictions'],
                                     predicted_boxes=inferred['total_detections'],
                                     epochs=completion['epochs'],training_seconds=completion['elapsed_seconds'],
                                     first_losses=completion['first_epoch_losses'],last_losses=completion['last_epoch_losses'],
                                     checkpoint=str(weight),checkpoint_sha256=completion['checkpoint_sha256']))
    overview.save(EXP/'review/overview_midpoints.jpg',quality=90)
    integrity=json.loads((EXP/'audit/source_integrity.json').read_text())
    current={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()}
    assert current==integrity['files'],'Original package changed'
    assert hashlib.sha256((ROOT/'data/server_marker_two_camera_train_v1.zip').read_bytes()).hexdigest()==integrity['archive_sha256']
    result=subprocess.run([__import__('sys').executable,str(DATA/'verify_package.py')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    (EXP/'audit/final_package_verification.log').write_text(result.stdout+result.stderr)
    # Retain only the final camera checkpoints. Smoke diagnostics remain under audit.
    for camera in ('F3E','F59'):
        out=EXP/camera
        smoke_out=EXP/'audit/smoke'/camera
        smoke_out.mkdir(parents=True,exist_ok=True)
        for directory in list(out.glob('*smoke*')):
            if directory.is_dir():
                shutil.move(str(directory),str(smoke_out/directory.name))
        for p in (out/'train/weights').glob('epoch*.pt'):
            p.unlink()  # generated intermediate snapshots, final last.pt has been verified above
        for p in smoke_out.rglob('*.pt'):
            p.unlink()  # completed one-epoch smoke checkpoint is superseded by the final models
    table(EXP/'review/group_summary.csv',group_summaries)
    dump(EXP/'summary.json',dict(status='completed',cameras=camera_summaries,qualitative_groups=group_summaries,
                               independent_validation=False,independent_test=False,original_source_unchanged=True,
                               total_training_images=378,total_training_boxes=483,total_qualitative_images=1946,
                               independent_accuracy_metrics=None,
                               reason='All reviewed serves 01/06 are train-only; other serves remain unlabeled.'))
    cleanup=json.loads((EXP/'cleanup_manifest.json').read_text())
    versions=json.loads((EXP/'environment_versions.json').read_text())
    lines=['最新双相机数据训练与测试推理报告','',
           'F3E、F59 已分别从官方 YOLO11n 初始化，完成独立模型训练及对应相机其他发球推理。',
           '本包没有独立标注验证集或测试集。定性推理不是定量泛化评估，不能报告 precision、recall 或 mAP。',
           '', '数据审计（包内 verify_package.py、全像素/标签/坐标对齐检查均通过）',
           '相机  标注帧  框  显式负样本  训练fallback  未标注推理帧  推理fallback']
    for s in camera_summaries:
        lines.append(f"{s['camera']:4s} {s['training_images']:6d} {s['training_boxes']:4d} {s['explicit_negatives']:10d} {s['training_fallback_frames']:12d} {s['unlabeled_frames']:12d} {s['unlabeled_fallback_frames']:12d}")
    lines+=['第 1、6 球所有已确认标注均训练；未随机拆分相邻帧。YAML val 指向 train 只是兼容字段。',
            'F3E 的源轨迹 manifest 有 368 个候选帧，其中 155 帧未确认；F59 有 218 个候选帧，其中 53 帧未确认。',
            '这些未确认帧未作为负样本或训练样本；定性推理图像来自另 7 个发球。',
            '全部图像 256×256，单通道 0/255。过滤图逐像素为未滤波图子集，极性图非零位置与未滤波图一致。',
            '随机训练 GT 抽样：audit/F3E_training_GT.jpg、audit/F59_training_GT.jpg。',
            'fallback 为轨迹未核验标记，不表示标注框无效。本次保留包中全部已确认样本，并在推理统计中单列。',
            '', '环境与训练设置',
            'Python 3.12.2；PyTorch 2.6.0+cu124；CUDA runtime 12.4；Ultralytics 8.4.192；RTX 3090。',
            'F3E 使用 GPU 5，F59 使用 GPU 7；各进程固定 2 个 CPU 线程。已有 GPU 任务未终止或修改。',
            '保留原有 CUDA/PyTorch。启动时修复了项目 NumPy 2 与 Matplotlib 二进制不兼容，固定 NumPy 1.26.4。',
            '依赖元数据与实际模块版本均有记录：environment_versions.json、environment_freeze.txt。',
            f"实际 OpenCV 模块版本：{versions['actual_cv2']}；已安装发行包元数据为 4.10.0.84，两者不一致，报告以实际导入记录为准。",
            '每台相机单独初始化官方 weights/yolo11n.pt；未使用旧实验训练权重。',
            '官方预训练来源：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt 。',
            '首层由 [16,3,3,3] 适配为 [16,1,3,3]，实测与 RGB 卷积核沿输入通道求和逐元素相等。',
            '实际训练冒烟批次 [4,1,256,256]，前向、反向、文件预测和完整一轮训练均成功。',
            '每个正式模型固定训练 150 轮，seed 42、deterministic=True、imgsz 256、batch 16、AdamW lr0 0.001、cosine、AMP。',
            '关闭颜色、旋转、平移、缩放、透视、翻转、mosaic、mixup、cutmix、copy-paste 和额外增强。',
            '冒烟实测训练输入值只有 0/255，模型输入除以 255；网络是浮点权重网络，不是二值网络。',
            '自定义 TrainOnlyTrainer 关闭常规验证、最终验证和验证驱动早停，CSV 仅写训练损失与学习率。',
            '没有 best.pt；最终模型 last.pt 为最终轮次的 EMA 权重，不经过独立验证选优。']
    for s in camera_summaries:
        lines += [f"{s['camera']}: {s['epochs']} 轮，用时 {s['training_seconds']:.1f} 秒；训练首轮损失 {s['first_losses']}；末轮损失 {s['last_losses']}。",
                  f"  模型：{s['camera']}/train/weights/last.pt；训练曲线：{s['camera']}/train/training_loss.png。"]
    lines+=['训练拟合自检：<相机>/training_fitcheck/contact_sheet_GT_green_prediction_cyan.jpg，绿色 GT、青色预测。',
            '训练损失下降和训练图匹配只反映拟合，不表示其他发球准确率。',
            '', '其他发球推理',
            '每台相机处理 serve02、03、04、05、07、08、09；合计 14 组 / 1946 帧，全部保留。',
            '模型与相机严格匹配；FP32，实际单通道 256×256；显式分块保证真实 batch 不超过 16。',
            '显示阈值 conf=0.25，保存候选下限 0.05，NMS IoU 0.7；未使用未标注数据调参。',
            '所有逐图预测均关联 manifest 原时间戳和 track_support；fallback 分组统计见 review/group_summary.csv。',
            '青色框全部为预测，不代表已确认正确；无框不代表已确认漏检。',
            '每组 overlays/*.jpg 与 review_slow.mp4 均按过滤预测图 / 未滤波图 / 极性图三列展示。',
            '视频采用 25 FPS，是 0.5 ms 导出图序列的 80 倍慢放；14 个视频逐一验证帧数完整。',
            '每组 reference_original.mp4 链接到包内原始双路复核视频，没有重复复制视频。',
            'frames.csv 保存原视频近邻帧估计：round((timestamp-video_first_us)/5000)。',
            '原复核视频播放器为 30 FPS，播放器定位秒数=该帧索引/30；不能用真实时间差直接作为播放器秒数。',
            '映射只是依据 metadata 的未核验估计，0.5 ms 图像与 5 ms 复核视频不是严格逐帧同步。',
            'manual_review.csv 预留人工审核字段；没有自动把预测转换为人工真值。',
            '', '分组预测统计（仅描述数量，不是准确率）',
            'group           frames  有框帧  无框帧  预测框  fallback帧']
    for g in group_summaries:
        lines.append(f"{g['group']:14s} {g['frames']:6d} {g['frames_with_predictions']:7d} {g['frames_without_predictions']:7d} {g['predicted_boxes']:7d} {g['fallback_frames']:10d}")
    lines+=['', '人工审核入口',
            'review/contact_sheets/：14 组各 6 帧时间等距抽样；review/overview_midpoints.jpg 每组固定中间帧。',
            '<相机>/qualitative/<发球>/review_slow.mp4：全部帧慢放。',
            '<相机>/qualitative/<发球>/predictions.json、frames.csv、detections.csv：全部框、置信度、时间定位。',
            '优先审查 fallback 样本、多框/无框片段及球缘附近的预测；是否误检或漏检须人工确认。',
            'F3E 的 serve08 有 55/119 帧 fallback，serve05 有 14/49，适合优先核验裁剪。',
            'F59 的 serve05 有 10/149 帧 fallback；建议跨发球、跨光照补充完全标注和真实负样本。',
            '只有每台相机两个发球的有限标注，且相邻样本高度相关，不能承诺真实新发球或其他条件下的准确率。',
            '未验证跨帧跟踪、三维球速或旋转速度。',
            '', '复现命令',
            'bash scripts/predict_latest.sh F3E /同相机过滤二值图路径 5',
            'bash scripts/predict_latest.sh F59 /同相机过滤二值图路径 7',
            '完整过程见 scripts/reproduce_latest.sh，完成的输出禁止覆盖；请在新工作目录复现。',
            '脚本、参数、运行环境和权重 SHA256 均已保存。',
            '', '目录清理',
            '活动工作目录仅保留最新数据、环境、官方预训练权重、最新双相机实验及脚本。',
            f"旧版本数据与旧实验已移到工作目录外：{cleanup['legacy_archive']} 。",
            '旧内容未永久删除，可按需恢复；本次实验仅保留两个最终正式模型，已清除本次中间及冒烟权重。',
            '最新原始 ZIP 和全部解压文件已在训练与推理结束后再次逐文件 SHA256 验证未变。',
            '', '许可',
            'Ultralytics 的 AGPL-3.0 文本保存为 ULTRALYTICS_LICENSE.txt；官方另提供 Enterprise 许可。',
            '本次本地训练与可视化未对外发布，数据未联网传输。后续使用和分发按适用许可执行。']
    (EXP/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'README.txt').write_text('当前仅使用最新双相机数据集\n\n数据：data/server_marker_two_camera_train_v1/\n原包：data/server_marker_two_camera_train_v1.zip\n完整报告：experiments/two_camera_v1/REPORT.txt\n预览图：experiments/two_camera_v1/review/contact_sheets/\nF3E 模型：experiments/two_camera_v1/F3E/train/weights/last.pt\nF59 模型：experiments/two_camera_v1/F59/train/weights/last.pt\n各相机测试推理：experiments/two_camera_v1/<相机>/qualitative/\n注意：本包无独立标注验证/测试集，测试推理仅供人工定性审核，不能计算泛化准确率。\n\n旧版本归档在：'+cleanup['legacy_archive']+'\n',encoding='utf-8')
    dump(EXP/'audit/output_verification.json',dict(status='passed',training_epochs_each=150,models=2,
                                                  qualitative_groups=14,qualitative_frames=1946,
                                                  source_unchanged=True,independent_metrics_not_computed=True,
                                                  training_csv_contains_loss_only=True,video_frame_counts_verified=True))
    keep={}
    for path in EXP.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.name not in ('deliverables_sha256.json','finalization.log') and 'working_data' not in path.parts and 'ultralytics_config' not in path.parts:
            keep[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(EXP/'deliverables_sha256.json',keep)
    print('COMPLETED',EXP)
    print(json.dumps(camera_summaries,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
