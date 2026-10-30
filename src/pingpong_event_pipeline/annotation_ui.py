"""Qt player for manual marker annotations synchronized with review video."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class AnnotationCanvas(QtWidgets.QLabel):
    def __init__(self, changed, pause):
        super().__init__()
        self.setFixedSize(512, 512)
        self.setMouseTracking(True)
        self.changed = changed
        self.pause = pause
        self.image = None
        self.boxes = []
        self.start = None
        self.current = None

    def set_image(self, image, boxes):
        self.image, self.boxes = image, boxes
        self.start = self.current = None
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if self.image is not None:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
            painter.drawImage(self.rect(), self.image)
        painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
        for x1, y1, x2, y2 in self.boxes:
            painter.drawRect(QtCore.QRectF(x1*2, y1*2, (x2-x1)*2, (y2-y1)*2))
        if self.start is not None and self.current is not None:
            painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
            painter.drawRect(QtCore.QRectF(self.start, self.current).normalized())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.pause()
            self.start = self.current = QtCore.QPointF(event.pos())

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.current = QtCore.QPointF(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None:
            return
        rect = QtCore.QRectF(self.start, QtCore.QPointF(event.pos())).normalized()
        x1, y1 = max(0., rect.left()/2), max(0., rect.top()/2)
        x2, y2 = min(256., rect.right()/2), min(256., rect.bottom()/2)
        if x2-x1 >= 2 and y2-y1 >= 2:
            self.boxes.append([x1, y1, x2, y2])
            self.changed()
        self.start = self.current = None
        self.update()


class Window(QtWidgets.QWidget):
    def __init__(self, group, video, video_first_us, video_step_us):
        super().__init__()
        self.group = group
        self.source = json.loads((group / 'source.json').read_text(encoding='utf-8'))
        with (group / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError('empty manifest')
        self.video = cv2.VideoCapture(str(video))
        if not self.video.isOpened():
            raise ValueError(f'cannot open video: {video}')
        self.video_first_us, self.video_step_us = video_first_us, video_step_us
        self.index = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.move(1))
        self.canvas = AnnotationCanvas(self.save, self.timer.stop)
        self.video_view = QtWidgets.QLabel('video')
        self.video_view.setFixedSize(1024, 288)
        self.reference_view = QtWidgets.QLabel('polarity')
        self.reference_view.setFixedSize(256, 256)
        self.status = QtWidgets.QLabel()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, len(self.rows)-1)
        self.slider.valueChanged.connect(self.goto)
        self.play_button = QtWidgets.QPushButton('播放 / 暂停 (Space)')
        self.play_button.clicked.connect(self.toggle)
        prev = QtWidgets.QPushButton('上一帧 (A)'); prev.clicked.connect(lambda: self.move(-1))
        nxt = QtWidgets.QPushButton('下一帧 (D)'); nxt.clicked.connect(lambda: self.move(1))
        undo = QtWidgets.QPushButton('撤销最后一框'); undo.clicked.connect(self.undo)
        negative = QtWidgets.QPushButton('确认无标记并保存'); negative.clicked.connect(self.negative)
        buttons = QtWidgets.QHBoxLayout()
        for b in (prev, self.play_button, nxt, undo, negative): buttons.addWidget(b)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.canvas)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.video_view)
        right.addWidget(QtWidgets.QLabel('极性参考：蓝 OFF / 黄 ON（仅供参照）'))
        right.addWidget(self.reference_view)
        body.addLayout(right)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('拖框后自动保存并纳入导出；请框完该帧所有标记。无标记须单独确认。'))
        layout.addLayout(body); layout.addWidget(self.status); layout.addWidget(self.slider); layout.addLayout(buttons)
        self.setWindowTitle('球面标记人工标注（未标注不视为负样本）')
        for key, action in [('Space', self.toggle), ('A', lambda: self.move(-1)), ('D', lambda: self.move(1))]:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
        self.goto(0)

    def image_qt(self, path):
        image = QtGui.QImage(str(path))
        if image.isNull(): raise ValueError(f'cannot open {path}')
        return image

    def goto(self, index):
        self.index = max(0, min(index, len(self.rows)-1))
        row = self.rows[self.index]
        image_path = self.group / row['image']
        annotation = image_path.with_suffix('.json')
        boxes = []
        review_state = '未标注'
        if annotation.exists():
            data = json.loads(annotation.read_text(encoding='utf-8'))
            review_state = '已确认完整' if data.get('flags', {}).get('reviewed') is True else '草稿（未确认）'
            for shape in data.get('shapes', []):
                if shape.get('label') != 'marker' or shape.get('shape_type') != 'rectangle':
                    raise ValueError(f'unsupported annotation: {annotation}')
                (x1, y1), (x2, y2) = shape['points']
                boxes.append([x1, y1, x2, y2])
        self.canvas.set_image(self.image_qt(image_path), boxes)
        ref = self.image_qt(self.group / row['polarity_reference'])
        self.reference_view.setPixmap(QtGui.QPixmap.fromImage(ref))
        video_index = round((int(row['timestamp_us']) - self.video_first_us) / self.video_step_us)
        if 0 <= video_index < int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, video_index)
            ok, frame = self.video.read()
            if ok:
                camera = self.source.get('camera', 'F3E' if self.group.name.endswith('F3E') else 'F59')
                x = round(float(row['center_x_px'])) + (1280 if camera == 'F59' else 0)
                y = round(float(row['center_y_px']))
                cv2.circle(frame, (x, y), max(1, round(float(row['radius_px']))), (0, 255, 0), 2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                q = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                self.video_view.setPixmap(QtGui.QPixmap.fromImage(q).scaled(1024, 288, QtCore.Qt.KeepAspectRatio))
            else:
                self.video_view.clear(); self.video_view.setText('原视频读取失败')
        else:
            self.video_view.clear(); self.video_view.setText('当前帧不在原视频范围内')
        self.slider.blockSignals(True); self.slider.setValue(self.index); self.slider.blockSignals(False)
        video_offset = (self.video_first_us + video_index*self.video_step_us - int(row['timestamp_us'])) / 1000
        support = row.get('track_support', '')
        tracking_note = (' | 警告：球心为回退候选，不能视为可靠追踪' if support == 'fallback_unverified'
                         else ' | 圆周支持为启发式，仍需核验球心' if support == 'ring_supported' else '')
        self.status.setText(f"帧 {self.index+1}/{len(self.rows)} | RAW {row['timestamp_us']} us | 视频帧 {video_index} (偏差 {video_offset:+.1f} ms) | "
                            + review_state + tracking_note)

    def save(self, reviewed=None):
        self.timer.stop()
        row = self.rows[self.index]
        image_path = self.group / row['image']
        shapes = [{'label': 'marker', 'points': [[x1,y1],[x2,y2]], 'group_id': None,
                   'shape_type': 'rectangle', 'flags': {}} for x1,y1,x2,y2 in self.canvas.boxes]
        if reviewed is None:
            reviewed = bool(shapes)
        payload = {'version': '5.5.0', 'flags': {'reviewed': reviewed}, 'shapes': shapes, 'imagePath': image_path.name,
                   'imageData': None, 'imageHeight': 256, 'imageWidth': 256}
        target = image_path.with_suffix('.json')
        tmp = target.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(target)
        self.goto(self.index)

    def negative(self):
        if QtWidgets.QMessageBox.question(self, '确认', '确认画面中没有可标注的标记点？这会清除本帧所有框。') == QtWidgets.QMessageBox.Yes:
            self.canvas.boxes = []; self.save(reviewed=True)

    def undo(self):
        if self.canvas.boxes:
            self.canvas.boxes.pop(); self.save()

    def toggle(self):
        if self.timer.isActive(): self.timer.stop()
        else: self.timer.start(100)

    def move(self, delta):
        if self.index + delta >= len(self.rows): self.timer.stop(); return
        self.goto(self.index + delta)

    def closeEvent(self, event):
        self.timer.stop(); self.video.release(); event.accept()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', required=True, type=Path)
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--video-first-us', required=True, type=int)
    p.add_argument('--video-step-us', type=int, default=5000)
    args = p.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = Window(args.group, args.video, args.video_first_us, args.video_step_us)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__': main()
