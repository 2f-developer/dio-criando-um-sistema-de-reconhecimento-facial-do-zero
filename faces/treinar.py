"""Treina o classificador de rostos.

python -m faces.treinar                 # com o LFW (baixa ~230 MB na primeira vez)
python -m faces.treinar --pasta fotos/  # com as suas fotos, uma subpasta por pessoa
"""

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from .dados import carregar_lfw, carregar_pasta, dividir
from .modelo import criar_classificador


def treinar(imagens, rotulos, nomes, pasta_modelo=Path("modelo"), pasta_resultados=Path("resultados"), epocas=(8, 6)):
    keras.utils.set_random_seed(42)
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = dividir(imagens, rotulos)
    # Peso maior para as fotos das pessoas com menos exemplos, para a classe maior não dominar.
    pesos_amostra = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)[y_tr].astype("float32")
    modelo, base = criar_classificador(len(nomes))

    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=epocas[0], batch_size=32, sample_weight=pesos_amostra, verbose=2)

    base.trainable = True
    for camada in base.layers[:-40]:
        camada.trainable = False
    modelo.compile(optimizer=keras.optimizers.Adam(2e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    parada = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    modelo.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=sum(epocas),
        initial_epoch=epocas[0],
        batch_size=32,
        sample_weight=pesos_amostra,
        callbacks=[parada],
        verbose=2,
    )

    pasta_modelo.mkdir(exist_ok=True)
    pasta_resultados.mkdir(exist_ok=True)
    modelo.save(pasta_modelo / "classificador_faces.keras")
    (pasta_modelo / "pessoas.json").write_text(json.dumps(nomes, ensure_ascii=False, indent=2))

    previstos = modelo.predict(x_te, verbose=0).argmax(axis=1)
    relatorio = classification_report(y_te, previstos, target_names=nomes, output_dict=True, zero_division=0)
    metricas = {
        "acuracia_teste": round(relatorio["accuracy"], 4),
        "f1_macro_teste": round(relatorio["macro avg"]["f1-score"], 4),
        "por_pessoa": {
            n: {
                "precisao": round(relatorio[n]["precision"], 3),
                "recall": round(relatorio[n]["recall"], 3),
                "fotos_teste": int(relatorio[n]["support"]),
            }
            for n in nomes
        },
        "fotos": {"treino": len(y_tr), "validacao": len(y_val), "teste": len(y_te)},
    }
    (pasta_resultados / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    _matriz(confusion_matrix(y_te, previstos), nomes, pasta_resultados / "matriz_confusao.png")
    np.savez_compressed(pasta_resultados / "teste.npz", imagens=x_te, rotulos=y_te)  # para a demonstração
    return metricas


def _matriz(matriz, nomes, destino):
    fig, eixo = plt.subplots(figsize=(7, 6))
    eixo.imshow(matriz, cmap="Blues")
    curtos = [n.split()[-1] for n in nomes]
    eixo.set_xticks(range(len(nomes)), curtos, rotation=45, ha="right")
    eixo.set_yticks(range(len(nomes)), curtos)
    for (i, j), valor in np.ndenumerate(matriz):
        eixo.text(j, i, valor, ha="center", va="center", color="white" if valor > matriz.max() / 2 else "black")
    eixo.set_xlabel("previsto")
    eixo.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(destino, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)  # GPUs pequenas (4 GB) não reservam tudo de uma vez
    parser = argparse.ArgumentParser(prog="python -m faces.treinar")
    parser.add_argument("--pasta", type=Path, help="fotos próprias, uma subpasta por pessoa")
    args = parser.parse_args(argv)
    dados = carregar_pasta(args.pasta) if args.pasta else carregar_lfw(Path("datasets"))
    print(json.dumps(treinar(*dados), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
