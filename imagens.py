"""Carregamento e pré-processamento das imagens.

Cada par de imagens pode vir de câmeras diferentes, com resoluções e
orientações EXIF diferentes. Este módulo garante que, a partir daqui, todo
o resto do código trabalhe sempre com imagens já corrigidas e na mesma
escala relativa (maior lado = config.MAIOR_LADO).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class ParCarregado:
    """Par de imagens já carregado, corrigido e redimensionado."""

    par: config.Par
    img1_bgr: np.ndarray
    img2_bgr: np.ndarray
    img1_cinza: np.ndarray
    img2_cinza: np.ndarray
    fator_escala1: float  # imagem_redimensionada = imagem_original * fator
    fator_escala2: float
    tamanho_original1: tuple[int, int]  # (largura, altura) antes de redimensionar
    tamanho_original2: tuple[int, int]


def carregar_imagem(caminho) -> np.ndarray:
    """Lê uma imagem do disco já com a orientação EXIF aplicada.

    O `cv2.imread` (OpenCV >= 3.1, com suporte a libjpeg/libexif) já aplica
    a tag EXIF "Orientation" ao decodificar o JPEG quando o flag padrão
    `IMREAD_COLOR` é usado, então as fotos tiradas em retrato no celular
    (20260312_*.jpg) chegam aqui já na orientação correta para exibição.
    """

    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho}")
    return imagem


def fator_de_escala(largura: int, altura: int) -> float:
    """Calcula o fator de redimensionamento para que o maior lado vire MAIOR_LADO.

    Se a imagem já é menor que MAIOR_LADO e a ampliação está desligada em
    config.py, o fator é 1.0 (a imagem fica do jeito que está).
    """

    maior_lado_atual = max(largura, altura)
    fator = config.MAIOR_LADO / maior_lado_atual
    if fator > 1.0 and not config.PERMITIR_AMPLIACAO:
        return 1.0
    return fator


def redimensionar(imagem: np.ndarray) -> tuple[np.ndarray, float]:
    """Redimensiona a imagem conforme `fator_de_escala` e devolve o fator usado."""

    altura, largura = imagem.shape[:2]
    fator = fator_de_escala(largura, altura)
    if fator == 1.0:
        return imagem, 1.0

    nova_largura = max(1, round(largura * fator))
    nova_altura = max(1, round(altura * fator))
    # INTER_AREA é a interpolação recomendada pela documentação do OpenCV
    # para reduzir imagens, por evitar aliasing.
    imagem_redim = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return imagem_redim, fator


def para_cinza(imagem_bgr: np.ndarray) -> np.ndarray:
    """Converte para escala de cinza, formato esperado pelos detectores."""

    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)


def escalar_pontos(pontos: np.ndarray, fator: float) -> np.ndarray:
    """Aplica o mesmo fator de redimensionamento a um conjunto de pontos (x, y).

    Usado para levar anotações feitas na imagem original para a escala de
    trabalho (MAIOR_LADO), já que a anotação é sempre salva em coordenadas
    da imagem original (ver anotacao.py).
    """

    pontos = np.asarray(pontos, dtype=np.float64)
    return pontos * fator


def mascara_do_poligono(forma: tuple[int, int], poligono: np.ndarray) -> np.ndarray:
    """Cria uma máscara binária (uint8, 0/255) com o interior do polígono preenchido.

    `forma` é (altura, largura). O polígono delimita a região do plano
    principal de cada cena -- paredes de fundo, céu, água e outros planos
    da cena ficam fora da máscara e são ignorados por detectores e métricas.
    """

    altura, largura = forma
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    pontos_int = np.round(np.asarray(poligono)).astype(np.int32)
    cv2.fillPoly(mascara, [pontos_int], color=255)
    return mascara


def carregar_par(par: config.Par) -> ParCarregado:
    """Carrega, corrige e redimensiona as duas imagens de um par."""

    caminho1 = config.PASTA_IMAGENS / par.imagem1
    caminho2 = config.PASTA_IMAGENS / par.imagem2

    bruta1 = carregar_imagem(caminho1)
    bruta2 = carregar_imagem(caminho2)

    tamanho_original1 = (bruta1.shape[1], bruta1.shape[0])  # (largura, altura)
    tamanho_original2 = (bruta2.shape[1], bruta2.shape[0])

    img1_bgr, fator1 = redimensionar(bruta1)
    img2_bgr, fator2 = redimensionar(bruta2)

    return ParCarregado(
        par=par,
        img1_bgr=img1_bgr,
        img2_bgr=img2_bgr,
        img1_cinza=para_cinza(img1_bgr),
        img2_cinza=para_cinza(img2_bgr),
        fator_escala1=fator1,
        fator_escala2=fator2,
        tamanho_original1=tamanho_original1,
        tamanho_original2=tamanho_original2,
    )
