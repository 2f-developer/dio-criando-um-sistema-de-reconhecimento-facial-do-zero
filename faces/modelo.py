"""Rede de classificação dos rostos: MobileNetV2 (ImageNet) + cabeça softmax com uma saída por pessoa."""

import keras
from keras import layers

from .dados import TAMANHO


def criar_classificador(n_pessoas: int, pesos: str | None = "imagenet") -> tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(input_shape=(TAMANHO, TAMANHO, 3), include_top=False, weights=pesos)
    base.trainable = False
    entrada = keras.Input((TAMANHO, TAMANHO, 3), name="rosto_rgb_0_255")
    x = layers.RandomFlip("horizontal")(entrada)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomContrast(0.15)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    saida = layers.Dense(n_pessoas, activation="softmax", name="pessoa")(x)
    return keras.Model(entrada, saida, name="reconhecimento_facial"), base
