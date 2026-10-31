"""Cálculo de descritores (SIFT e ORB) para qualquer lista de keypoints.

O ponto delicado deste módulo: keypoints vêm de três origens possíveis
(Harris, SIFT, ORB) mas qualquer um dos dois descritores (SIFT ou ORB)
precisa conseguir descrevê-los -- é isso que a questão 2 pede (todas as
combinações detector x descritor).

O campo `octave` de um cv2.KeyPoint tem um significado específico para
cada algoritmo: o SIFT empacota aí o índice da oitava E da camada da
pirâmide de escala-espaço em bits diferentes do mesmo inteiro, e o ORB usa
esse campo para o nível da pirâmide de resolução do ORB. Se entregarmos a
um descritor um keypoint com o `octave` "no formato" de outro detector, o
descritor decodifica esse número errado e tenta ler a imagem numa oitava
que não existe -- na prática isso derruba pontos silenciosamente ou lança
exceção. Por isso, antes de descrever um keypoint que não foi nativamente
criado pelo próprio detector/descritor, zeramos o `octave` (mantendo
posição, tamanho e ângulo) para que o descritor sempre trabalhe na
resolução cheia da imagem.
"""

from __future__ import annotations

import cv2
import numpy as np

DetectorNativo = str  # "Harris" | "SIFT" | "ORB"


def _preparar_keypoints(keypoints: list[cv2.KeyPoint], origem: DetectorNativo, descritor: str) -> list[cv2.KeyPoint]:
    """Zera `octave` quando o keypoint não é nativo do descritor que vai descrevê-lo."""

    if origem == descritor:
        return list(keypoints)

    preparados = []
    for kp in keypoints:
        novo = cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, 0, kp.class_id)
        preparados.append(novo)
    return preparados


def descrever_sift(
    cinza: np.ndarray, keypoints: list[cv2.KeyPoint], origem: DetectorNativo
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Computa descritores SIFT (128-D, float32) para os keypoints dados.

    `compute()` pode descartar keypoints muito perto da borda da imagem
    (não há vizinhança suficiente para montar o descritor), então a lista
    de keypoints devolvida pode ser menor que a de entrada -- isso é
    esperado e registrado por quem chama esta função.
    """

    sift = cv2.SIFT_create()
    preparados = _preparar_keypoints(keypoints, origem, "SIFT")
    kps_mantidos, descritores = sift.compute(cinza, preparados)
    if descritores is None:
        descritores = np.empty((0, 128), dtype=np.float32)
    return list(kps_mantidos), descritores


def descrever_orb(
    cinza: np.ndarray, keypoints: list[cv2.KeyPoint], origem: DetectorNativo
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Computa descritores ORB (256 bits binários, uint8) para os keypoints dados."""

    orb = cv2.ORB_create()
    preparados = _preparar_keypoints(keypoints, origem, "ORB")
    kps_mantidos, descritores = orb.compute(cinza, preparados)
    if descritores is None:
        descritores = np.empty((0, 32), dtype=np.uint8)
    return list(kps_mantidos), descritores


# Registro: nome do descritor -> (função, norma usada no matching).
# NORM_L2 é a distância euclidiana, adequada a descritores reais (SIFT).
# NORM_HAMMING conta bits diferentes, adequada a descritores binários (ORB).
DESCRITORES = {
    "SIFT": (descrever_sift, cv2.NORM_L2),
    "ORB": (descrever_orb, cv2.NORM_HAMMING),
}
