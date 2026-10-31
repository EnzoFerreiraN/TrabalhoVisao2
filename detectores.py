"""Detectores de pontos de interesse (POIs): Harris, SIFT e ORB.

Todas as funções seguem a mesma assinatura -- recebem a imagem em escala de
cinza, um dicionário de parâmetros e uma máscara opcional -- e devolvem uma
lista de `cv2.KeyPoint`. Isso permite que `main.py` trate os três
detectores de forma intercambiável (mesmo "formato de encaixe").
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

import config

DetectorFn = Callable[[np.ndarray, dict, np.ndarray | None], list[cv2.KeyPoint]]


def detectar_harris(
    cinza: np.ndarray,
    params: dict = config.PADRAO_HARRIS,
    mascara: np.ndarray | None = None,
) -> list[cv2.KeyPoint]:
    """Detecta cantos com o operador de Harris.

    O Harris (ao contrário de SIFT/ORB) não devolve "keypoints" prontos,
    apenas um mapa de resposta por pixel. Por isso aqui fazemos nós mesmos
    as duas etapas que faltam: (1) limiarização relativa ao máximo da
    resposta e (2) supressão de não-máximos (NMS), para não ter uma nuvem
    de pixels vizinhos todos marcados como "cantos" no mesmo canto físico.
    """

    cinza_f32 = np.float32(cinza)
    resposta = cv2.cornerHarris(
        cinza_f32,
        blockSize=params["blockSize"],
        ksize=config.HARRIS_KSIZE,
        k=params["k"],
    )

    if mascara is not None:
        resposta = resposta.copy()
        resposta[mascara == 0] = 0.0

    limiar_absoluto = params["limiar"] * resposta.max()
    if limiar_absoluto <= 0:
        return []

    # Supressão de não-máximos: um pixel só sobrevive se for o máximo da
    # sua própria vizinhança (janela HARRIS_JANELA_NMS x HARRIS_JANELA_NMS).
    dilatada = cv2.dilate(resposta, np.ones((config.HARRIS_JANELA_NMS, config.HARRIS_JANELA_NMS), np.uint8))
    mascara_maximos = (resposta == dilatada) & (resposta > limiar_absoluto)

    ys, xs = np.nonzero(mascara_maximos)
    keypoints = [
        cv2.KeyPoint(float(x), float(y), size=config.HARRIS_TAMANHO_KEYPOINT, response=float(resposta[y, x]))
        for y, x in zip(ys, xs)
    ]
    return keypoints


def detectar_sift(
    cinza: np.ndarray,
    params: dict = config.PADRAO_SIFT,
    mascara: np.ndarray | None = None,
) -> list[cv2.KeyPoint]:
    """Detecta keypoints com SIFT (escala e orientação nativas)."""

    sift = cv2.SIFT_create(
        contrastThreshold=params["contrastThreshold"],
        edgeThreshold=params["edgeThreshold"],
    )
    return list(sift.detect(cinza, mascara))


def detectar_orb(
    cinza: np.ndarray,
    params: dict = config.PADRAO_ORB,
    mascara: np.ndarray | None = None,
) -> list[cv2.KeyPoint]:
    """Detecta keypoints com ORB (FAST + orientação por centroide de intensidade).

    `nfeatures` é um TETO: o ORB ordena os cantos encontrados pela resposta
    do FAST e devolve no máximo essa quantidade, então aumentar o valor só
    tem efeito até o ORB "esgotar" os cantos de fato detectados na imagem.
    """

    orb = cv2.ORB_create(
        nfeatures=params["nfeatures"],
        fastThreshold=params["fastThreshold"],
    )
    return list(orb.detect(cinza, mascara))


DETECTORES: dict[str, DetectorFn] = {
    "Harris": detectar_harris,
    "SIFT": detectar_sift,
    "ORB": detectar_orb,
}

PARAMS_PADRAO: dict[str, dict] = {
    "Harris": config.PADRAO_HARRIS,
    "SIFT": config.PADRAO_SIFT,
    "ORB": config.PADRAO_ORB,
}

VARIACOES: dict[str, dict] = {
    "Harris": config.VARIACOES_HARRIS,
    "SIFT": config.VARIACOES_SIFT,
    "ORB": config.VARIACOES_ORB,
}
