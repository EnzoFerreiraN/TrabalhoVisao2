"""Homografia de referência (ground truth) de cada par de imagens.

IMPORTANTE: a homografia usada para medir repetibilidade e taxa de acerto
NUNCA vem do RANSAC dos próprios casamentos SIFT/ORB -- isso seria
circular (estaríamos usando o resultado do algoritmo para avaliar o
próprio algoritmo). Em vez disso, a homografia vem de 8 a 12 pontos
correspondentes marcados manualmente (ver anotacao.py), o que dá uma
referência independente dos detectores/descritores testados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from imagens import ParCarregado, escalar_pontos, mascara_do_poligono


@dataclass
class Anotacao:
    """Conteúdo bruto do JSON de anotação de um par (coordenadas originais)."""

    roi_imagem1: np.ndarray  # (N, 2) polígono da ROI, na imagem 1 original
    pontos_imagem1: np.ndarray  # (M, 2) pontos correspondentes, imagem 1 original
    pontos_imagem2: np.ndarray  # (M, 2) pontos correspondentes, imagem 2 original


@dataclass
class ReferenciaPar:
    """Tudo que as questões 1 e 2 precisam sobre a geometria de um par.

    Todas as coordenadas/máscaras já estão na escala de trabalho (a mesma
    escala das imagens redimensionadas em `imagens.carregar_par`).
    """

    homografia: np.ndarray  # H tal que H @ p1_homogeneo ~= p2_homogeneo
    erro_medio_px: float
    mascara1: np.ndarray  # ROI do plano principal, na imagem 1
    mascara2: np.ndarray  # ROI do plano principal, projetada na imagem 2
    roi1: np.ndarray
    roi2: np.ndarray
    homografia_manual: np.ndarray | None = None  # H só dos cliques (antes do ECC)
    ajuste_ecc_px: float = 0.0  # quanto o ECC moveu a projeção (mediana na ROI, px)


def caminho_anotacao(par: config.Par, pasta: Path = config.PASTA_ANOTACOES) -> Path:
    return pasta / f"{par.nome}.json"


def anotacao_existe(par: config.Par, pasta: Path = config.PASTA_ANOTACOES) -> bool:
    return caminho_anotacao(par, pasta).exists()


def carregar_anotacao(par: config.Par, pasta: Path = config.PASTA_ANOTACOES) -> Anotacao:
    """Lê o JSON salvo por `anotacao.py` para o par informado."""

    caminho = caminho_anotacao(par, pasta)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Anotação não encontrada para o par '{par.nome}' ({caminho}).\n"
            f"Rode `python anotacao.py --par {par.nome}` antes de continuar."
        )

    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    return Anotacao(
        roi_imagem1=np.array(dados["roi_imagem1"], dtype=np.float64),
        pontos_imagem1=np.array(dados["pontos_imagem1"], dtype=np.float64),
        pontos_imagem2=np.array(dados["pontos_imagem2"], dtype=np.float64),
    )


def calcular_homografia(pontos1: np.ndarray, pontos2: np.ndarray) -> np.ndarray:
    """Calcula H a partir de pontos correspondentes (mínimos quadrados por padrão).

    Usa o método configurado em config.HOMOGRAFIA_METODO. Com poucos pontos
    marcados manualmente (8-12), o padrão é usar todos eles (método 0),
    já que a curadoria de qual ponto é "inlier" aqui é feita pelo próprio
    usuário ao clicar com cuidado -- ver o erro de reprojeção impresso por
    `anotacao.py` para validar isso.
    """

    metodo = config.HOMOGRAFIA_METODO
    kwargs = {}
    if metodo == cv2.RANSAC:
        kwargs["ransacReprojThreshold"] = config.HOMOGRAFIA_LIMIAR_RANSAC

    H, _ = cv2.findHomography(pontos1, pontos2, method=metodo, **kwargs)
    if H is None:
        raise RuntimeError("cv2.findHomography não conseguiu calcular uma homografia.")
    return H


def projetar_pontos(pontos: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Projeta pontos (N, 2) por uma homografia (3, 3), devolvendo (N, 2)."""

    pontos = np.asarray(pontos, dtype=np.float64).reshape(-1, 1, 2)
    projetados = cv2.perspectiveTransform(pontos, H)
    return projetados.reshape(-1, 2)


def erro_reprojecao(H: np.ndarray, pontos1: np.ndarray, pontos2: np.ndarray) -> np.ndarray:
    """Distância euclidiana (px) entre H @ p1 e p2, ponto a ponto."""

    projetados = projetar_pontos(pontos1, H)
    return np.linalg.norm(projetados - np.asarray(pontos2, dtype=np.float64), axis=1)



def refinar_homografia_ecc(
    img1_cinza: np.ndarray, img2_cinza: np.ndarray, H_inicial: np.ndarray, mascara2: np.ndarray
) -> np.ndarray:
    """Refina H (imagem 1 -> imagem 2) por alinhamento ECC, do nível grosso ao fino.

    O ECC procura a homografia W tal que img2(W·x) ~ img1(x) para os pixels
    x do plano, maximizando o coeficiente de correlação entre as
    intensidades. Ele precisa de um bom chute inicial (a H manual) e é
    invariante a mudanças de brilho/contraste entre as fotos. Na convenção
    do OpenCV, W leva coordenadas do "template" (img1) para o "input"
    (img2), ou seja, W já é a H 1->2 que queremos. A máscara indica os
    pixels válidos do input (a ROI projetada na imagem 2).
    """

    H = H_inicial.astype(np.float32) / np.float32(H_inicial[2, 2])
    criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config.ECC_MAX_ITERACOES, config.ECC_EPSILON)

    for nivel in reversed(range(config.ECC_NIVEIS_PIRAMIDE)):
        escala = 1.0 / (2 ** nivel)
        S = np.diag([escala, escala, 1.0]).astype(np.float32)
        S_inv = np.diag([1 / escala, 1 / escala, 1.0]).astype(np.float32)

        t1 = cv2.resize(img1_cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA).astype(np.float32)
        t2 = cv2.resize(img2_cinza, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA).astype(np.float32)
        m2 = cv2.resize(mascara2, (t2.shape[1], t2.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Homografia expressa nas coordenadas deste nível da pirâmide.
        H_nivel = S @ H @ S_inv
        _, H_nivel = cv2.findTransformECC(
            t1, t2, H_nivel, cv2.MOTION_HOMOGRAPHY, criterio, m2, config.ECC_TAMANHO_FILTRO_GAUSS
        )
        H = S_inv @ H_nivel @ S

    return (H / H[2, 2]).astype(np.float64)


def deslocamento_entre_homografias(H_a: np.ndarray, H_b: np.ndarray, mascara1: np.ndarray, passo: int = 10) -> np.ndarray:
    """Distância (px) entre as projeções de H_a e H_b em uma grade de pontos da ROI da imagem 1."""

    ys, xs = np.nonzero(mascara1[::passo, ::passo])
    pontos = np.column_stack([xs * passo, ys * passo]).astype(np.float64)
    return np.linalg.norm(projetar_pontos(pontos, H_a) - projetar_pontos(pontos, H_b), axis=1)


def obter_referencia(par_carregado: ParCarregado, pasta: Path = config.PASTA_ANOTACOES) -> ReferenciaPar:
    """Monta a homografia e as máscaras de ROI na escala de trabalho do par.

    A anotação é sempre feita e salva em coordenadas da imagem ORIGINAL
    (ver anotacao.py), para que trocar `config.MAIOR_LADO` não invalide o
    trabalho manual já feito. Aqui escalamos tudo para a resolução em que
    o par foi efetivamente carregado.
    """

    anotacao = carregar_anotacao(par_carregado.par, pasta)

    pontos1 = escalar_pontos(anotacao.pontos_imagem1, par_carregado.fator_escala1)
    pontos2 = escalar_pontos(anotacao.pontos_imagem2, par_carregado.fator_escala2)
    roi1 = escalar_pontos(anotacao.roi_imagem1, par_carregado.fator_escala1)

    H_manual = calcular_homografia(pontos1, pontos2)
    mascara1 = mascara_do_poligono(par_carregado.img1_cinza.shape, roi1)

    H = H_manual
    ajuste_ecc_px = 0.0
    if config.REFINAR_HOMOGRAFIA_ECC:
        mascara2_manual = mascara_do_poligono(par_carregado.img2_cinza.shape, projetar_pontos(roi1, H_manual))
        H = refinar_homografia_ecc(par_carregado.img1_cinza, par_carregado.img2_cinza, H_manual, mascara2_manual)
        ajuste_ecc_px = float(np.median(deslocamento_entre_homografias(H_manual, H, mascara1)))

    # Erro dos cliques manuais em relação à H final (indica a precisão da anotação).
    erros = erro_reprojecao(H, pontos1, pontos2)

    roi2 = projetar_pontos(roi1, H)
    mascara2 = mascara_do_poligono(par_carregado.img2_cinza.shape, roi2)

    return ReferenciaPar(
        homografia=H,
        erro_medio_px=float(np.mean(erros)),
        mascara1=mascara1,
        mascara2=mascara2,
        roi1=roi1,
        roi2=roi2,
        homografia_manual=H_manual,
        ajuste_ecc_px=ajuste_ecc_px,
    )
