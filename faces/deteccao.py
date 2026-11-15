"""Detecção de faces com a YuNet (OpenCV), uma rede leve que roda bem em CPU.

O modelo (Apache 2.0, do projeto opencv_zoo) é baixado na primeira execução se não estiver em modelo/.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

URL_YUNET = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
CAMINHO_YUNET = Path(__file__).resolve().parent.parent / "modelo" / "face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    largura: int
    altura: int
    confianca: float


class Detector:
    def __init__(self, limiar: float = 0.8, caminho: Path = CAMINHO_YUNET) -> None:
        if not caminho.exists():
            caminho.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL_YUNET, caminho)
        self._yunet = cv2.FaceDetectorYN.create(str(caminho), "", (320, 320), limiar, 0.3, 5000)

    def detectar(self, imagem_bgr: np.ndarray) -> list[Face]:
        altura, largura = imagem_bgr.shape[:2]
        self._yunet.setInputSize((largura, altura))
        _, resultado = self._yunet.detect(imagem_bgr)
        if resultado is None:
            return []
        return [Face(int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[14])) for r in resultado]


def recortar(imagem: np.ndarray, face: Face, margem: float = 0.25) -> np.ndarray:
    """Recorte quadrado em volta do rosto, com folga, sem sair da imagem."""
    lado = int(max(face.largura, face.altura) * (1 + 2 * margem))
    cx, cy = face.x + face.largura // 2, face.y + face.altura // 2
    altura, largura = imagem.shape[:2]
    x1, y1 = max(0, cx - lado // 2), max(0, cy - lado // 2)
    x2, y2 = min(largura, x1 + lado), min(altura, y1 + lado)
    return imagem[y1:y2, x1:x2]
