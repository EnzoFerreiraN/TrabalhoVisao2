"""Casamento (matching) de descritores entre as duas imagens de um par.

Usa força bruta (BFMatcher) com k=2 vizinhos mais próximos e o teste de
razão de Lowe para filtrar casamentos ambíguos, como recomendado no artigo
original do SIFT.
"""

from __future__ import annotations

import cv2
import numpy as np

import config


def casar(descritores1: np.ndarray, descritores2: np.ndarray, norma: int) -> list[cv2.DMatch]:
    """Casa descritores com BFMatcher(norma) + kNN(k=2) + razão de Lowe.

    Um casamento só é aceito se a distância ao melhor vizinho for menor que
    `config.RAZAO_LOWE` vezes a distância ao segundo melhor vizinho --
    isso descarta casamentos ambíguos (quando duas regiões da imagem 2 são
    parecidas o suficiente com a região da imagem 1 para confundir o
    matcher).
    """

    if len(descritores1) == 0 or len(descritores2) < 2:
        return []

    matcher = cv2.BFMatcher(norma)
    pares_candidatos = matcher.knnMatch(descritores1, descritores2, k=2)

    aceitos = []
    for candidatos in pares_candidatos:
        # Com poucos descritores em descritores2, pode faltar o segundo
        # vizinho -- nesse caso não há como aplicar o teste de razão.
        if len(candidatos) < 2:
            continue
        melhor, segundo = candidatos
        if melhor.distance < config.RAZAO_LOWE * segundo.distance:
            aceitos.append(melhor)

    return aceitos


def classificar_casamentos(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    casamentos: list[cv2.DMatch],
    homografia: np.ndarray,
    epsilon_px: float = config.EPSILON_PX,
) -> np.ndarray:
    """Marca cada casamento como correto/incorreto segundo a homografia de referência.

    Um casamento (p1, p2) é CORRETO se a distância entre H @ p1 e p2 for
    menor que `epsilon_px`. H vem da anotação manual (homografia.py), nunca
    do próprio matching -- ver o aviso em homografia.py sobre circularidade.
    """

    from homografia import projetar_pontos

    if not casamentos:
        return np.zeros(0, dtype=bool)

    pontos1 = np.array([keypoints1[m.queryIdx].pt for m in casamentos], dtype=np.float64)
    pontos2 = np.array([keypoints2[m.trainIdx].pt for m in casamentos], dtype=np.float64)

    projetados = projetar_pontos(pontos1, homografia)
    distancias = np.linalg.norm(projetados - pontos2, axis=1)
    return distancias < epsilon_px
