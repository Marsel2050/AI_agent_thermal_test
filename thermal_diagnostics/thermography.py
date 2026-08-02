from __future__ import annotations

import io
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class ThermogramError(RuntimeError):
    pass


class ExifToolMissingError(ThermogramError):
    pass


class NonRadiometricImageError(ThermogramError):
    pass


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def bounded(self, image_width: int, image_height: int) -> "Region":
        x = min(max(self.x, 0), image_width - 1)
        y = min(max(self.y, 0), image_height - 1)
        width = min(max(self.width, 1), image_width - x)
        height = min(max(self.height, 1), image_height - y)
        return Region(x, y, width, height)


@dataclass(frozen=True)
class RegionStatistics:
    minimum: float
    maximum: float
    mean: float
    median: float
    percentile_95: float


@dataclass
class ThermogramData:
    matrix_celsius: np.ndarray
    metadata: dict

    @property
    def width(self) -> int:
        return int(self.matrix_celsius.shape[1])

    @property
    def height(self) -> int:
        return int(self.matrix_celsius.shape[0])


def region_statistics(matrix: np.ndarray, region: Region) -> RegionStatistics:
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("Ожидается непустая двумерная матрица температур")
    bounded = region.bounded(matrix.shape[1], matrix.shape[0])
    roi = matrix[
        bounded.y : bounded.y + bounded.height,
        bounded.x : bounded.x + bounded.width,
    ]
    finite = roi[np.isfinite(roi)]
    if finite.size == 0:
        raise ValueError("В выбранной области нет корректных температур")
    return RegionStatistics(
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        percentile_95=float(np.percentile(finite, 95)),
    )


def thermal_preview(
    matrix: np.ndarray,
    contact_region: Region | None = None,
    reference_region: Region | None = None,
) -> Image.Image:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Матрица не содержит корректных температур")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((matrix - low) / (high - low), 0, 1)
    try:
        from matplotlib import colormaps

        rgb = colormaps["inferno"](normalized, bytes=True)[..., :3]
    except ImportError:
        gray = (normalized * 255).astype(np.uint8)
        rgb = np.stack((gray, gray, gray), axis=-1)
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for region, color in ((contact_region, "#00ff66"), (reference_region, "#00b7ff")):
        if region is None:
            continue
        bounded = region.bounded(image.width, image.height)
        draw.rectangle(
            (
                bounded.x,
                bounded.y,
                bounded.x + bounded.width - 1,
                bounded.y + bounded.height - 1,
            ),
            outline=color,
            width=max(1, min(image.width, image.height) // 100),
        )
    return image


def read_preview(content: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise ThermogramError(f"Не удалось открыть изображение: {exc}") from exc


def read_radiometric_flir(content: bytes, filename: str) -> ThermogramData:
    """Extract a Celsius matrix from a radiometric FLIR/DJI JPEG."""

    if shutil.which("exiftool") is None:
        raise ExifToolMissingError(
            "ExifTool не найден. Установите libimage-exiftool-perl или ExifTool for Windows."
        )
    try:
        from flirimageextractor import FlirImageExtractor
    except ImportError as exc:
        raise ThermogramError("Не установлен пакет flirimageextractor") from exc

    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    try:
        extractor = FlirImageExtractor(exiftool_path="exiftool")
        if not extractor.check_for_thermal_image(str(temporary_path)):
            raise NonRadiometricImageError(
                "Файл не содержит RawThermalImage. Цветная картинка не является матрицей температур."
            )
        metadata = extractor.get_metadata(str(temporary_path))
        extractor.process_image(str(temporary_path))
        matrix = np.asarray(extractor.get_thermal_np(), dtype=np.float64)
        if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).any():
            raise ThermogramError("Из файла не извлечена корректная матрица температур")
        return ThermogramData(matrix_celsius=matrix, metadata=metadata)
    finally:
        temporary_path.unlink(missing_ok=True)
