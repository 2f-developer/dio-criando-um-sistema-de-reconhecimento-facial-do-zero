import numpy as np

from faces.dados import TAMANHO, dividir, para_uint8, preparar_rosto
from faces.deteccao import Face, recortar
from faces.modelo import criar_classificador


def test_recorte_quadrado_com_margem_e_dentro_da_imagem():
    imagem = np.zeros((100, 200, 3), dtype=np.uint8)
    assert recortar(imagem, Face(80, 30, 40, 40, 0.9), margem=0.25).shape[:2] == (60, 60)
    canto = recortar(imagem, Face(0, 0, 40, 40, 0.9), margem=0.5)
    assert canto.shape[0] <= 100 and canto.shape[1] <= 200


def test_preparar_rosto_gera_rgb_no_tamanho_do_modelo():
    bgr = np.zeros((90, 70, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # azul em BGR
    rgb = preparar_rosto(bgr)
    assert rgb.shape == (TAMANHO, TAMANHO, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]


def test_divisao_estratificada_sem_vazamento():
    imagens = np.arange(200).reshape(200, 1, 1, 1)
    rotulos = np.repeat([0, 1, 2, 3], 50)
    (xt, yt), (xv, yv), (xs, ys) = dividir(imagens, rotulos)
    ids = [set(x.ravel()) for x in (xt, xv, xs)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(yt) == 140 and len(yv) == 30 and len(ys) == 30
    assert np.bincount(ys).tolist() == [7, 8, 8, 7] or set(np.bincount(ys)) <= {7, 8}


def test_classificador_tem_uma_saida_por_pessoa_e_soma_um():
    modelo, base = criar_classificador(5, pesos=None)
    saida = modelo.predict(np.random.default_rng(0).uniform(0, 255, (2, TAMANHO, TAMANHO, 3)), verbose=0)
    assert saida.shape == (2, 5)
    np.testing.assert_allclose(saida.sum(axis=1), 1, rtol=1e-5)
    assert not base.trainable


def test_para_uint8_aceita_float_de_0_a_1_e_inteiro_de_0_a_255():
    assert para_uint8(np.array([0.0, 0.5, 1.0], dtype=np.float32)).tolist() == [0, 128, 255]
    assert para_uint8(np.array([0, 128, 255], dtype=np.uint8)).tolist() == [0, 128, 255]
