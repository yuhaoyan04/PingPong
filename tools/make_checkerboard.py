"""生成 A4、9x6 内角点、20 mm 方格的双目标定板。请按 100% 比例打印。"""
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    dpi, square_mm = 300, 20
    mm_to_px = dpi / 25.4
    page = Image.new("RGB", (round(210 * mm_to_px), round(297 * mm_to_px)), "white")
    draw = ImageDraw.Draw(page)
    square = round(square_mm * mm_to_px)
    cols, rows = 10, 7  # 9x6 internal corners
    ox = (page.width - cols * square) // 2
    oy = (page.height - rows * square) // 2
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                draw.rectangle((ox + x * square, oy + y * square,
                                ox + (x + 1) * square - 1, oy + (y + 1) * square - 1), fill="black")
    output = Path("calibration")
    output.mkdir(exist_ok=True)
    page.save(output / "checkerboard_9x6_20mm_A4.pdf", "PDF", resolution=dpi)
    page.save(output / "checkerboard_9x6_20mm_A4.png", dpi=(dpi, dpi))
    print((output / "checkerboard_9x6_20mm_A4.pdf").resolve())


if __name__ == "__main__":
    main()
