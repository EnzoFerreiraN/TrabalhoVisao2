"""Métricas da questão 1 (repetibilidade, sobreposição) e da questão 2 (taxa de acerto)."""

from __future__ import annotations

import cv2
import numpy as np

import config
from homografia import projetar_pontos


def _pontos_de(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    if not keypoints:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in keypoints], dtype=np.float64)


def distancia_ao_vizinho_mais_proximo(pontos_a: np.ndarray, pontos_b: np.ndarray) -> np.ndarray:
    """Para cada ponto de `pontos_a`, a distância euclidiana ao ponto mais próximo em `pontos_b`.

    Calculado em blocos (chunks) para não montar uma matriz de distâncias
    gigante de uma vez quando há muitos milhares de pontos.
    """

    n = len(pontos_a)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if len(pontos_b) == 0:
        return np.full(n, np.inf, dtype=np.float64)

    TAMANHO_BLOCO = 512
    distancias_minimas = np.empty(n, dtype=np.float64)
    for inicio in range(0, n, TAMANHO_BLOCO):
        fim = min(inicio + TAMANHO_BLOCO, n)
        bloco = pontos_a[inicio:fim]  # (b, 2)
        diffs = bloco[:, None, :] - pontos_b[None, :, :]  # (b, m, 2)
        dists = np.sqrt((diffs**2).sum(axis=2))  # (b, m)
        distancias_minimas[inicio:fim] = dists.min(axis=1)
    return distancias_minimas


def pontos_dentro_da_imagem(pontos_int: np.ndarray, largura: int, altura: int) -> np.ndarray:
    """Máscara booleana: quais pontos (já arredondados para inteiro) caem dentro de (0, largura) x (0, altura).

    Recebe pontos JÁ ARREDONDADOS, não as coordenadas float originais: um
    ponto com, por exemplo, y=479.6 passa no teste com limite float
    (479.6 < 480), mas `np.round` leva ele a y=480, que já está fora da
    imagem (índices válidos vão de 0 a altura-1). Arredondar antes de
    checar os limites evita esse off-by-one e o IndexError correspondente.
    """

    return (
        (pontos_int[:, 0] >= 0)
        & (pontos_int[:, 0] < largura)
        & (pontos_int[:, 1] >= 0)
        & (pontos_int[:, 1] < altura)
    )


def pontos_dentro_da_mascara(pontos: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Máscara booleana: quais pontos (x, y) caem em pixels != 0 da máscara da ROI."""

    altura, largura = mascara.shape
    pontos_int = np.round(pontos).astype(np.int64)
    dentro_imagem = pontos_dentro_da_imagem(pontos_int, largura, altura)
    resultado = np.zeros(len(pontos), dtype=bool)
    if not np.any(dentro_imagem):
        return resultado

    indices = np.nonzero(dentro_imagem)[0]
    xs = pontos_int[indices, 0]
    ys = pontos_int[indices, 1]
    resultado[indices] = mascara[ys, xs] != 0
    return resultado


def repetibilidade(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    homografia: np.ndarray,
    mascara2: np.ndarray,
    epsilon_px: float = config.EPSILON_PX,
) -> tuple[float, int, int]:
    """Fração de pontos da imagem 1 que "repetem" na imagem 2.

    Um ponto de `keypoints1` repete se, projetado por `homografia`, cai
    dentro da ROI da imagem 2 (mascara2) E existe algum ponto de
    `keypoints2` a até `epsilon_px` dele.

    Pontos que caem fora da ROI projetada são ignorados na contagem
    (não é possível repetir algo que nem está na região comum às duas
    vistas), o que evita penalizar o detector por conta da geometria da
    cena, e não do algoritmo em si.

    Devolve (fracao_repetibilidade, n_pontos_considerados, n_repetidos).
    """

    pontos1 = _pontos_de(keypoints1)
    pontos2 = _pontos_de(keypoints2)

    if len(pontos1) == 0:
        return 0.0, 0, 0

    projetados = projetar_pontos(pontos1, homografia)
    validos = pontos_dentro_da_mascara(projetados, mascara2)
    n_considerados = int(np.sum(validos))
    if n_considerados == 0:
        return 0.0, 0, 0

    distancias = distancia_ao_vizinho_mais_proximo(projetados[validos], pontos2)
    n_repetidos = int(np.sum(distancias <= epsilon_px))

    return n_repetidos / n_considerados, n_considerados, n_repetidos


def sobreposicao(
    keypoints_a: list[cv2.KeyPoint], keypoints_b: list[cv2.KeyPoint], epsilon_px: float = config.EPSILON_PX
) -> tuple[float, int]:
    """Fração dos pontos de A que têm algum ponto de B a até `epsilon_px` de distância.

    Mede o quanto dois detectores concordam sobre ONDE estão os pontos de
    interesse na mesma imagem (não sobre a descrição deles).
    Devolve (fracao_sobreposicao, n_pontos_em_a).
    """

    pontos_a = _pontos_de(keypoints_a)
    pontos_b = _pontos_de(keypoints_b)

    if len(pontos_a) == 0:
        return 0.0, 0

    distancias = distancia_ao_vizinho_mais_proximo(pontos_a, pontos_b)
    fracao = float(np.mean(distancias <= epsilon_px))
    return fracao, len(pontos_a)


def taxa_acerto(n_corretos: int, n_total: int) -> float:
    """Fração de casamentos corretos, protegida contra divisão por zero."""

    if n_total == 0:
        return 0.0
    return n_corretos / n_total
