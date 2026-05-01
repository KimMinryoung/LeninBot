#!/usr/bin/env python3
"""Generate nonogram puzzle sheets and pixel-art solutions from source images."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    raise SystemExit("opencv-python-headless is required for nonogram image binarization") from exc

from nonogram.src.core.solveurs.dynamic_programming import solve as nonogram_solve
from nonogram.src.core.solveurs.solveur_utils import CASE_NOIRE, CASE_VIDE


CELL_SIZE = 40
FONT_SIZE = 18
OUTPUT_DIR = Path("output")
PUZZLES_DIR = Path("puzzles")

SOYUZ_PALETTE: list[tuple[int, int, int]] = [
    (255, 255, 255),  # paper
    (204, 32, 39),  # poster red
    (31, 83, 177),  # cobalt blue
    (242, 187, 51),  # warm yellow
    (28, 31, 36),  # ink black
]

COLOR_NAMES = {
    (255, 255, 255): "paper",
    (204, 32, 39): "poster_red",
    (31, 83, 177): "cobalt_blue",
    (242, 187, 51): "warm_yellow",
    (28, 31, 36): "ink_black",
}


@dataclass(frozen=True)
class PuzzleData:
    name: str
    title: str
    caption: str
    size: int
    colors: list[list[tuple[int, int, int]]]
    filled: list[list[bool]]
    row_clues: list[list[int]]
    column_clues: list[list[int]]
    background_color: tuple[int, int, int]
    solver_fully_resolved: bool
    solver_matches_source: bool


def crop_center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def crop_largest_subject_square(image: Image.Image) -> Image.Image:
    rgb_image = composite_on_white(image)
    rgb = np.array(rgb_image, dtype=np.uint8)
    foreground = (np.linalg.norm(255 - rgb.astype(np.int16), axis=2) > 40).astype(np.uint8)
    if not np.any(foreground):
        return crop_center_square(rgb_image)

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    if labels_count <= 1:
        return crop_center_square(rgb_image)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    margin = round(max(width, height) * 0.22)
    center_x = x + width / 2
    center_y = y + height / 2
    side = min(max(width, height) + margin * 2, max(rgb_image.size))
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    right = left + side
    bottom = top + side

    padded = Image.new("RGB", (side, side), (255, 255, 255))
    crop = rgb_image.crop((max(0, left), max(0, top), min(rgb_image.width, right), min(rgb_image.height, bottom)))
    padded.paste(crop, (max(0, -left), max(0, -top)))
    return padded


def composite_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def fixed_palette_image() -> Image.Image:
    palette = Image.new("P", (1, 1))
    flat_palette: list[int] = []
    for color in SOYUZ_PALETTE:
        flat_palette.extend(color)
    flat_palette.extend([0, 0, 0] * (256 - len(SOYUZ_PALETTE)))
    palette.putpalette(flat_palette)
    return palette


def quantize_to_soyuz_palette(image: Image.Image, size: int) -> Image.Image:
    square = crop_center_square(composite_on_white(image))
    resized = square.resize((size, size), Image.Resampling.LANCZOS)
    quantized = resized.quantize(palette=fixed_palette_image(), dither=Image.Dither.NONE)
    return quantized.convert("RGB")


def amplify_lab_channel(channel: np.ndarray, scale: float) -> np.ndarray:
    centered = channel.astype(np.float32) - 128.0
    amplified = 128.0 + centered * scale
    return np.clip(amplified, 0, 255).astype(np.uint8)


def nearest_palette_color(rgb: np.ndarray) -> tuple[int, int, int]:
    palette = np.array(SOYUZ_PALETTE, dtype=np.int32)
    delta = palette - rgb.astype(np.int32)
    index = int(np.argmin(np.sum(delta * delta, axis=1)))
    return SOYUZ_PALETTE[index]


def _binarize_image(image: Image.Image, size: int) -> tuple[list[list[bool]], np.ndarray, np.ndarray]:
    center_square = crop_center_square(composite_on_white(image))
    if min(image.size) > size * 2:
        square = crop_largest_subject_square(image)
    else:
        square = center_square
    source_side = square.size[0]
    work_size = max(size * CELL_SIZE, 256)
    resampling = Image.Resampling.NEAREST if source_side <= size * 2 else Image.Resampling.LANCZOS
    resized = square.resize((work_size, work_size), resampling)
    rgb = np.array(resized, dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab_amplified = lab.copy()
    lab_amplified[:, :, 1] = amplify_lab_channel(lab[:, :, 1], 1.8)
    lab_amplified[:, :, 2] = amplify_lab_channel(lab[:, :, 2], 1.8)

    amplified_rgb = cv2.cvtColor(lab_amplified, cv2.COLOR_LAB2RGB)
    grayscale = cv2.cvtColor(amplified_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        4,
    )
    edges = cv2.Canny(blurred, 55, 135)
    edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)

    merged = np.maximum(adaptive, edges)
    if source_side <= size * 2:
        foreground = (np.linalg.norm(255 - rgb.astype(np.int16), axis=2) > 40).astype(np.uint8) * 255
        merged = foreground
    reduced = Image.fromarray(merged, mode="L").resize((size, size), resampling)
    reduced_array = np.array(reduced, dtype=np.uint8)
    return (reduced_array >= 28).tolist(), rgb, merged


def quantize_cells_from_mask(
    rgb: np.ndarray,
    detail_mask: np.ndarray,
    filled: list[list[bool]],
) -> list[list[tuple[int, int, int]]]:
    size = len(filled)
    height, width = detail_mask.shape
    colors: list[list[tuple[int, int, int]]] = []
    for row in range(size):
        y0 = round(row * height / size)
        y1 = round((row + 1) * height / size)
        color_row: list[tuple[int, int, int]] = []
        for column in range(size):
            if not filled[row][column]:
                color_row.append(SOYUZ_PALETTE[0])
                continue

            x0 = round(column * width / size)
            x1 = round((column + 1) * width / size)
            cell_mask = detail_mask[y0:y1, x0:x1] > 0
            cell_rgb = rgb[y0:y1, x0:x1]
            visible_mask = np.linalg.norm(255 - cell_rgb.astype(np.int16), axis=2) > 40
            if np.any(visible_mask):
                mean_rgb = cell_rgb[visible_mask].mean(axis=0)
            elif np.any(cell_mask):
                mean_rgb = cell_rgb[cell_mask].mean(axis=0)
            else:
                mean_rgb = cell_rgb.reshape(-1, 3).mean(axis=0)

            color = nearest_palette_color(mean_rgb)
            if color == SOYUZ_PALETTE[0]:
                color = SOYUZ_PALETTE[4]
            color_row.append(color)
        colors.append(color_row)
    return colors


def border_pixels(colors: list[list[tuple[int, int, int]]]) -> Iterable[tuple[int, int, int]]:
    last = len(colors) - 1
    for index, color in enumerate(colors[0]):
        yield color
    if last > 0:
        for color in colors[last]:
            yield color
    for row in colors[1:last]:
        yield row[0]
        yield row[last]


def detect_background_color(colors: list[list[tuple[int, int, int]]]) -> tuple[int, int, int]:
    counts = Counter(border_pixels(colors))
    if counts:
        return counts.most_common(1)[0][0]
    return SOYUZ_PALETTE[0]


def clues_for_line(cells: list[bool]) -> list[int]:
    clues: list[int] = []
    run = 0
    for cell in cells:
        if cell:
            run += 1
        elif run:
            clues.append(run)
            run = 0
    if run:
        clues.append(run)
    return clues


def solve_with_nonogram(row_clues: list[list[int]], column_clues: list[list[int]], source: list[list[bool]]) -> tuple[bool, bool]:
    grid = np.full((len(row_clues), len(column_clues)), CASE_VIDE)
    solved_grid, _elapsed = nonogram_solve(row_clues, column_clues, grid=grid)
    fully_resolved = bool(np.all(solved_grid != CASE_VIDE))
    if not fully_resolved:
        return False, False

    expected = np.array(source, dtype=int)
    solved = (solved_grid == CASE_NOIRE).astype(int)
    return True, bool(np.array_equal(solved, expected))


def build_puzzle_data(input_path: Path, name: str, size: int, title: str, caption: str) -> PuzzleData:
    if size < 2 or size > 50:
        raise ValueError("--size must be between 2 and 50")

    with Image.open(input_path) as image:
        filled, color_source, detail_mask = _binarize_image(image, size)

    colors = quantize_cells_from_mask(color_source, detail_mask, filled)
    background_color = SOYUZ_PALETTE[0]
    row_clues = [clues_for_line(row) for row in filled]
    column_clues = [clues_for_line([filled[row][column] for row in range(size)]) for column in range(size)]
    solver_fully_resolved, solver_matches_source = solve_with_nonogram(row_clues, column_clues, filled)

    return PuzzleData(
        name=name,
        title=title,
        caption=caption,
        size=size,
        colors=colors,
        filled=filled,
        row_clues=row_clues,
        column_clues=column_clues,
        background_color=background_color,
        solver_fully_resolved=solver_fully_resolved,
        solver_matches_source=solver_matches_source,
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def clue_layout(data: PuzzleData, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int, int, int]:
    scratch = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(scratch)
    digit_width, digit_height = text_size(draw, font, str(data.size))
    slot_width = max(digit_width + 10, 24)
    slot_height = max(digit_height + 8, 24)
    max_row_clues = max(1, max((len(clues) for clues in data.row_clues), default=1))
    max_column_clues = max(1, max((len(clues) for clues in data.column_clues), default=1))
    left = max_row_clues * slot_width + 12
    top = max_column_clues * slot_height + 12
    return left, top, slot_width, slot_height


def draw_grid_lines(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    size: int,
    cell_size: int,
    color: tuple[int, int, int],
) -> None:
    grid_size = size * cell_size
    for index in range(size + 1):
        offset = index * cell_size
        draw.line((left + offset, top, left + offset, top + grid_size), fill=color, width=1)
        draw.line((left, top + offset, left + grid_size, top + offset), fill=color, width=1)


def draw_clues(
    draw: ImageDraw.ImageDraw,
    data: PuzzleData,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    left: int,
    top: int,
    slot_width: int,
    slot_height: int,
) -> None:
    black = (0, 0, 0)
    for row_index, clues in enumerate(data.row_clues):
        display_clues = clues or [0]
        y_center = top + row_index * CELL_SIZE + CELL_SIZE // 2
        start_x = left - len(display_clues) * slot_width
        for clue_index, clue in enumerate(display_clues):
            text = str(clue)
            width, height = text_size(draw, font, text)
            x = start_x + clue_index * slot_width + slot_width - width - 5
            y = y_center - height // 2 - 1
            draw.text((x, y), text, fill=black, font=font)

    for column_index, clues in enumerate(data.column_clues):
        display_clues = clues or [0]
        x_center = left + column_index * CELL_SIZE + CELL_SIZE // 2
        start_y = top - len(display_clues) * slot_height
        for clue_index, clue in enumerate(display_clues):
            text = str(clue)
            width, height = text_size(draw, font, text)
            x = x_center - width // 2
            y = start_y + clue_index * slot_height + slot_height - height - 5
            draw.text((x, y), text, fill=black, font=font)


def render_puzzle(data: PuzzleData, output_path: Path, solution: bool) -> None:
    font = load_font(FONT_SIZE)
    left, top, slot_width, slot_height = clue_layout(data, font)
    width = left + data.size * CELL_SIZE
    height = top + data.size * CELL_SIZE
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw_clues(draw, data, font, left, top, slot_width, slot_height)

    if solution:
        for row in range(data.size):
            for column in range(data.size):
                if data.filled[row][column]:
                    x0 = left + column * CELL_SIZE + 1
                    y0 = top + row * CELL_SIZE + 1
                    x1 = x0 + CELL_SIZE - 2
                    y1 = y0 + CELL_SIZE - 2
                    draw.rectangle((x0, y0, x1, y1), fill=data.colors[row][column])
        draw_grid_lines(draw, left, top, data.size, CELL_SIZE, (115, 115, 115))
    else:
        draw_grid_lines(draw, left, top, data.size, CELL_SIZE, (0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_metadata(data: PuzzleData, output_path: Path, source_path: Path) -> None:
    payload = {
        "puzzle_id": data.name,
        "grid_size": data.size,
        "row_clues": data.row_clues,
        "column_clues": data.column_clues,
        "title": data.title,
        "caption": data.caption,
        "source_image": str(source_path),
        "palette": [{"name": COLOR_NAMES.get(color, "mapped"), "rgb": list(color)} for color in SOYUZ_PALETTE],
        "background_color": list(data.background_color),
        "filled_cells": sum(sum(1 for cell in row if cell) for row in data.filled),
        "solver": {
            "package": "nonogram",
            "fully_resolved": data.solver_fully_resolved,
            "matches_source": data.solver_matches_source,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_outputs(input_path: Path, name: str, size: int, title: str, caption: str) -> PuzzleData:
    data = build_puzzle_data(input_path, name, size, title, caption)
    render_puzzle(data, OUTPUT_DIR / f"{name}_puzzle.png", solution=False)
    render_puzzle(data, OUTPUT_DIR / f"{name}_solution.png", solution=True)
    write_metadata(data, OUTPUT_DIR / f"{name}_meta.json", input_path)
    return data


def save_matrix_image(name: str, matrix: list[str], color: tuple[int, int, int] = SOYUZ_PALETTE[1]) -> Path:
    size = len(matrix)
    image = Image.new("RGB", (size, size), SOYUZ_PALETTE[0])
    pixels = image.load()
    for y, row in enumerate(matrix):
        if len(row) != size:
            raise ValueError(f"{name}: test matrix must be square")
        for x, value in enumerate(row):
            if value != ".":
                pixels[x, y] = color
    path = PUZZLES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_star_image() -> Path:
    source_size = 160
    center = source_size / 2
    outer = source_size * 0.42
    inner = source_size * 0.18
    points: list[tuple[float, float]] = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((center + radius * math.cos(angle), center + radius * math.sin(angle)))

    image = Image.new("RGB", (source_size, source_size), SOYUZ_PALETTE[0])
    draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=SOYUZ_PALETTE[1])
    path = PUZZLES_DIR / "test_star.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def create_test_sources() -> list[tuple[Path, str, int, str, str]]:
    hammer_sickle = save_matrix_image(
        "test_hammer_sickle",
        [
            "RRRRRRRRR.",
            "...RR.....",
            "...RR.....",
            "...RR..RRR",
            "...RR.R..R",
            "...RR.R...",
            "...RR.R...",
            "...RR.R..R",
            "...RR..RRR",
            ".........R",
        ],
    )
    arrow_down = save_matrix_image(
        "test_arrow_down",
        [
            "..R..",
            "..R..",
            "RRRRR",
            ".RRR.",
            "..R..",
        ],
        color=SOYUZ_PALETTE[3],
    )
    star = save_star_image()
    return [
        (hammer_sickle, "test_hammer_sickle", 10, "망치와 낫", "노동의 상징을 10칸 픽셀 퍼즐로 압축했다."),
        (arrow_down, "test_arrow_down", 5, "하락 화살표", "이윤율 저하 경향을 위한 5칸 테스트 도상."),
        (star, "test_star", 8, "붉은 별", "작은 격자에서 정치적 상징이 어떻게 살아나는지 확인한다."),
    ]


def generate_tests() -> list[PuzzleData]:
    return [generate_outputs(*args) for args in create_test_sources()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate nonogram puzzle and solution PNGs from a source image.")
    parser.add_argument("--input", type=Path, help="Source PNG image path.")
    parser.add_argument("--name", help="Output puzzle id/name prefix.")
    parser.add_argument("--size", type=int, default=12, help="Target grid size, for example 5, 10, 12, 15, or 20.")
    parser.add_argument("--title", default="", help="Puzzle title for metadata.")
    parser.add_argument("--caption", default="", help="One-line caption for metadata.")
    parser.add_argument("--make-tests", action="store_true", help="Create and render the three built-in test puzzles.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.make_tests:
        generated = generate_tests()
        for data in generated:
            print(
                f"{data.name}: {data.size}x{data.size}, "
                f"filled={sum(sum(row) for row in data.filled)}, "
                f"solver_fully_resolved={data.solver_fully_resolved}, "
                f"solver_matches_source={data.solver_matches_source}"
            )
        return 0

    missing = [flag for flag, value in (("--input", args.input), ("--name", args.name)) if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    if not args.input.exists():
        raise SystemExit(f"input image not found: {args.input}")

    data = generate_outputs(args.input, args.name, args.size, args.title, args.caption)
    print(
        f"{data.name}: wrote {OUTPUT_DIR / (data.name + '_puzzle.png')}, "
        f"{OUTPUT_DIR / (data.name + '_solution.png')}, {OUTPUT_DIR / (data.name + '_meta.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
