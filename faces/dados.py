"""Fontes de rostos para treino: o LFW (Labeled Faces in the Wild) ou uma pasta sua.

Pasta própria: uma subpasta por pessoa, com as fotos dentro
    fotos/
      ana/  1.jpg 2.jpg ...
      bruno/  ...
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from .deteccao import Detector, recortar

TAMANHO = 160


def preparar_rosto(rosto_bgr: np.ndarray) -> np.ndarray:
    """Recorte do rosto → RGB 160x160, pixels 0 a 255 (a normalização fica dentro do modelo)."""
    return cv2.resize(cv2.cvtColor(rosto_bgr, cv2.COLOR_BGR2RGB), (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)


def para_uint8(imagem: np.ndarray) -> np.ndarray:
    """Converte para pixels inteiros de 0 a 255, aceitando imagens em float de 0 a 1."""
    if imagem.dtype != np.uint8 and imagem.max() <= 1.0:
        imagem = imagem * 255
    return np.clip(np.rint(imagem), 0, 255).astype(np.uint8)


def carregar_lfw(pasta_cache: Path, min_fotos: int = 60, max_por_pessoa: int = 150, semente: int = 42):
    """Pessoas do LFW com pelo menos `min_fotos` fotos. Limita o máximo por pessoa para a classe
    mais frequente (George W. Bush tem 530 fotos) não dominar o treino."""
    from sklearn.datasets import fetch_lfw_people

    lfw = fetch_lfw_people(
        data_home=str(pasta_cache),
        min_faces_per_person=min_fotos,
        color=True,
        resize=1.0,
        slice_=(slice(50, 200), slice(50, 200)),
        download_if_missing=True,
    )
    rng = np.random.default_rng(semente)
    indices = np.concatenate([rng.permutation(np.flatnonzero(lfw.target == c))[:max_por_pessoa] for c in range(len(lfw.target_names))])
    # O scikit-learn entrega pixels de 0 a 1; o modelo espera 0 a 255, como nas fotos da câmera.
    imagens = np.stack([cv2.resize(para_uint8(lfw.images[i]), (TAMANHO, TAMANHO)) for i in indices])
    return imagens, lfw.target[indices], [str(n) for n in lfw.target_names]


def carregar_pasta(pasta: Path, detector: Detector | None = None):
    """Lê fotos de uma pasta por pessoa, recortando o maior rosto de cada foto."""
    detector = detector or Detector()
    imagens, rotulos, nomes = [], [], sorted(p.name for p in pasta.iterdir() if p.is_dir())
    for indice, nome in enumerate(nomes):
        for arquivo in sorted((pasta / nome).iterdir()):
            foto = cv2.imread(str(arquivo))
            faces = detector.detectar(foto) if foto is not None else []
            if faces:
                maior = max(faces, key=lambda f: f.largura * f.altura)
                imagens.append(preparar_rosto(recortar(foto, maior)))
                rotulos.append(indice)
    if not imagens:
        raise ValueError(f"Nenhum rosto encontrado em {pasta}.")
    return np.stack(imagens), np.array(rotulos), nomes


def dividir(imagens: np.ndarray, rotulos: np.ndarray, semente: int = 42):
    """Treino 70%, validação 15%, teste 15%, mantendo a proporção de cada pessoa."""
    x_treino, x_resto, y_treino, y_resto = train_test_split(imagens, rotulos, test_size=0.3, stratify=rotulos, random_state=semente)
    x_val, x_teste, y_val, y_teste = train_test_split(x_resto, y_resto, test_size=0.5, stratify=y_resto, random_state=semente)
    return (x_treino, y_treino), (x_val, y_val), (x_teste, y_teste)
