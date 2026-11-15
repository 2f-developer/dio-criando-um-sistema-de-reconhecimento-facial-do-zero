"""Encontra e identifica todos os rostos de uma imagem.

    python -m faces.reconhecer FOTO [FOTO ...] [--saida pasta] [--limiar 0.6]

Rostos cuja confiança fica abaixo do limiar aparecem como "desconhecido": a rede só conhece
as pessoas do treino, e é melhor admitir isso do que chutar o nome mais parecido.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import keras
import numpy as np

from .dados import preparar_rosto
from .deteccao import Detector, Face, recortar


@dataclass(frozen=True)
class Reconhecimento:
    face: Face
    nome: str
    confianca: float


class Reconhecedor:
    def __init__(self, pasta_modelo: Path = Path("modelo"), limiar: float = 0.6) -> None:
        self.modelo = keras.models.load_model(pasta_modelo / "classificador_faces.keras")
        self.nomes = json.loads((pasta_modelo / "pessoas.json").read_text())
        self.detector = Detector()
        self.limiar = limiar

    def reconhecer(self, imagem_bgr: np.ndarray) -> list[Reconhecimento]:
        faces = self.detector.detectar(imagem_bgr)
        if not faces:
            return []
        lote = np.stack([preparar_rosto(recortar(imagem_bgr, f)) for f in faces])
        probabilidades = self.modelo.predict(lote, verbose=0)
        resultado = []
        for face, prob in zip(faces, probabilidades, strict=True):
            melhor = int(prob.argmax())
            nome = self.nomes[melhor] if prob[melhor] >= self.limiar else "desconhecido"
            resultado.append(Reconhecimento(face, nome, float(prob[melhor])))
        return resultado


def desenhar(imagem: np.ndarray, reconhecimentos: list[Reconhecimento]) -> np.ndarray:
    saida = imagem.copy()
    for r in reconhecimentos:
        f = r.face
        cor = (60, 180, 75) if r.nome != "desconhecido" else (60, 60, 220)
        cv2.rectangle(saida, (f.x, f.y), (f.x + f.largura, f.y + f.altura), cor, 2)
        texto = f"{r.nome} {r.confianca:.0%}"
        cv2.putText(saida, texto, (f.x, max(14, f.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)
    return saida


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m faces.reconhecer")
    parser.add_argument("fotos", nargs="+", type=Path)
    parser.add_argument("--saida", type=Path, default=Path("reconhecidos"))
    parser.add_argument("--limiar", type=float, default=0.6)
    args = parser.parse_args(argv)
    reconhecedor = Reconhecedor(limiar=args.limiar)
    args.saida.mkdir(parents=True, exist_ok=True)
    for foto in args.fotos:
        imagem = cv2.imread(str(foto))
        if imagem is None:
            print(f"{foto}: não consegui abrir")
            continue
        resultado = reconhecedor.reconhecer(imagem)
        cv2.imwrite(str(args.saida / f"{foto.stem}_faces.jpg"), desenhar(imagem, resultado))
        print(f"{foto.name}: " + (", ".join(f"{r.nome} ({r.confianca:.0%})" for r in resultado) or "nenhum rosto"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
