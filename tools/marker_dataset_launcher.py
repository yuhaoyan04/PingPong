"""Select an exported group and open its synchronized annotation player."""
import json
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtWidgets


def create_window(root):
    window = QtWidgets.QWidget()
    window.setWindowTitle('选择发球和相机：自动球心未经人工核验')
    layout = QtWidgets.QVBoxLayout(window)
    layout.addWidget(QtWidgets.QLabel('请选择待标注组。先检查球心裁剪和视频一致性；不可辨认的画面不要保存负样本。'))
    dataset_combo = QtWidgets.QComboBox()
    dataset_combo.addItem('原工作集（保留现有标注）', str(root/'output'/'marker_annotation_all_v1'))
    dataset_combo.addItem('改进轨迹候选（新工作集，未验证，标注不继承）', str(root/'output'/'marker_annotation_refined_v2'))
    dataset_combo.addItem('RAW 事件追踪候选（独立工作集，待复核）', str(root/'output'/'marker_annotation_event_v1'))
    dataset_combo.addItem('双相机增强新工作集（分相机去噪，标注不继承）', str(root/'output'/'marker_annotation_camera_enhanced_v1'))
    layout.addWidget(dataset_combo)
    combo = QtWidgets.QComboBox()
    layout.addWidget(combo)
    button = QtWidgets.QPushButton('打开播放标注窗口')
    def open_group():
        source = Path(combo.currentData())
        data = json.loads(source.read_text(encoding='utf-8'))
        subprocess.Popen([sys.executable, str(root/'tools'/'marker_annotation_player.py'),
                          '--group', str(source.parent), '--video', data['video'],
                          '--video-first-us', str(data['video_first_us']), '--video-step-us', str(data['video_step_us'])], cwd=str(root))
    button.clicked.connect(open_group)
    def refresh():
        combo.clear()
        dataset = Path(dataset_combo.currentData())
        for source in sorted(dataset.glob('*/source.json')):
            data = json.loads(source.read_text(encoding='utf-8'))
            if data.get('frames', 0) and (source.parent/'manifest.csv').is_file():
                fallback = data.get('fallback_frames', 0)
                suffix = f' — 回退 {fallback} 帧需重点检查' if fallback else ''
                combo.addItem(f"{source.parent.name} — {data['frames']} 帧 — 自动轨迹未核验{suffix}", str(source))
        button.setEnabled(combo.count() > 0)
    dataset_combo.currentIndexChanged.connect(refresh)
    refresh()
    layout.addWidget(button)
    window.resize(720, 150)
    return window


def main():
    root = Path(__file__).resolve().parents[1]
    app = QtWidgets.QApplication(sys.argv)
    window = create_window(root)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__': main()
