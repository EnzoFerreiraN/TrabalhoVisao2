"""Configuração central do projeto.

Tudo que é "número mágico" ou caminho de arquivo fica aqui, para que o
resto do código nunca precise ser mexido para ajustar um parâmetro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent
PASTA_IMAGENS = RAIZ / "imagens"
PASTA_ANOTACOES = RAIZ / "anotacoes"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"


# ---------------------------------------------------------------------------
# Pares de imagens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Par:
    """Um par de imagens da mesma cena, tiradas de pontos de vista próximos."""

    nome: str
    imagem1: str
    imagem2: str


# Os dois pares fornecidos pela disciplina + o par próprio (fachada em
# mosaico de Veneza). São os pares "principais", exigidos pelo enunciado.
PARES_PRINCIPAIS: list[Par] = [
    Par("dsc07631_dsc07632", "dsc07631.jpg", "dsc07632.jpg"),
    Par("dsc07654_dsc07655", "dsc07654.jpg", "dsc07655.jpg"),
    Par("veneza_121422_121424", "20260312_121422.jpg", "20260312_121424.jpg"),
]

# Pares extras: os mesmos pontos de vista dos pares "dsc", mas com frames
# mais afastados (menos sobreposição/deformação). Os pares dsc originais
# são frames consecutivos e quase idênticos, então os resultados tendem a
# saturar (repetibilidade e taxa de acerto muito altas); estes pares extras
# servem para comparar no relatório um caso com deformação maior entre as
# duas vistas. Ficam desligados por padrão para não dobrar o tempo de
# execução sempre que não forem necessários.
PARES_EXTRAS: list[Par] = [
    Par("dsc07631_dsc07635", "dsc07631.jpg", "dsc07635.jpg"),
    Par("dsc07654_dsc07658", "dsc07654.jpg", "dsc07658.jpg"),
]

INCLUIR_PARES_EXTRAS = False


def pares_ativos(incluir_extras: bool = INCLUIR_PARES_EXTRAS) -> list[Par]:
    """Devolve a lista de pares que o `main.py` deve processar."""

    pares = list(PARES_PRINCIPAIS)
    if incluir_extras:
        pares += PARES_EXTRAS
    return pares


# ---------------------------------------------------------------------------
# Pré-processamento das imagens
# ---------------------------------------------------------------------------

# Lado maior (em pixels) para o qual as imagens são redimensionadas antes de
# qualquer processamento. Isso torna os pares comparáveis entre si (as fotos
# da disciplina e as fotos próprias têm resoluções muito diferentes) e deixa
# o processamento rápido.
MAIOR_LADO = 1000

# Só reduzimos imagens maiores que MAIOR_LADO; imagens menores (como os
# pares "dsc", que já são 640x480) NÃO são ampliadas, porque ampliar não
# acrescenta informação real à imagem, apenas interpola pixels, o que
# mudaria artificialmente o comportamento dos detectores.
PERMITIR_AMPLIACAO = False


# ---------------------------------------------------------------------------
# Anotação manual (ground truth da homografia)
# ---------------------------------------------------------------------------

MIN_PONTOS_ANOTADOS = 8
MAX_PONTOS_ANOTADOS = 12

# Método do cv2.findHomography usado sobre os pontos marcados manualmente.
# 0 = mínimos quadrados comuns (todos os pontos pesam igual). Como os
# pontos são escolhidos à mão e em número pequeno, não faz sentido rodar
# RANSAC "de verdade" aqui -- isso é só para permitir trocar o método caso
# o usuário queira robustez a 1 ou 2 cliques ruins.
HOMOGRAFIA_METODO = 0
HOMOGRAFIA_LIMIAR_RANSAC = 3.0  # usado apenas se HOMOGRAFIA_METODO = cv2.RANSAC

# Refinamento da homografia manual por alinhamento ECC (Evangelidis &
# Psarakis, 2008). Os cliques manuais têm imprecisão de 1-4 px (pior nas
# fotos grandes, que aparecem encolhidas na janela de anotação) e, com só
# 8-12 pontos, a H pode extrapolar mal nas bordas da ROI. O ECC parte da H
# manual e ajusta os 8 parâmetros maximizando a correlação das INTENSIDADES
# dos pixels dentro da ROI -- não usa Harris, SIFT nem ORB, então a
# referência continua independente dos métodos avaliados.
REFINAR_HOMOGRAFIA_ECC = True
ECC_NIVEIS_PIRAMIDE = 3        # alinha do grosso (1/4) para o fino (1/1)
ECC_MAX_ITERACOES = 300
ECC_EPSILON = 1e-7
ECC_TAMANHO_FILTRO_GAUSS = 5


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

# Distância máxima (em pixels, na imagem redimensionada) para considerar que
# um ponto projetado "repetiu" ou que um casamento está correto.
EPSILON_PX = 3.0

# Semente fixa para todas as etapas com aleatoriedade (amostragem de
# casamentos desenhados, etc.), para resultados reproduzíveis.
SEMENTE = 42


# ---------------------------------------------------------------------------
# Detector: Harris
# ---------------------------------------------------------------------------

# O Harris não produz keypoints com escala/orientação como SIFT/ORB; para
# que ele possa ser descrito por SIFT/ORB na questão 2, cada canto detectado
# vira um cv2.KeyPoint com este tamanho fixo (diâmetro, em pixels).
HARRIS_TAMANHO_KEYPOINT = 16.0

# Tamanho da janela (em pixels) usada na supressão de não-máximos aplicada
# sobre o mapa de resposta do Harris.
HARRIS_JANELA_NMS = 5

# Abertura do operador de Sobel usado internamente pelo cornerHarris.
HARRIS_KSIZE = 3

PADRAO_HARRIS = {"k": 0.04, "blockSize": 3, "limiar": 0.01}

VARIACOES_HARRIS = {
    "k": [0.04, 0.05, 0.06],
    "blockSize": [2, 3, 5],
    "limiar": [0.005, 0.01, 0.05],
}


# ---------------------------------------------------------------------------
# Detector/descritor: SIFT
# ---------------------------------------------------------------------------

PADRAO_SIFT = {"contrastThreshold": 0.04, "edgeThreshold": 10}

VARIACOES_SIFT = {
    "contrastThreshold": [0.02, 0.04, 0.08],
    "edgeThreshold": [5, 10, 20],
}


# ---------------------------------------------------------------------------
# Detector/descritor: ORB
# ---------------------------------------------------------------------------

PADRAO_ORB = {"nfeatures": 1000, "fastThreshold": 20}

VARIACOES_ORB = {
    "nfeatures": [500, 1000, 3000],
    "fastThreshold": [10, 20, 40],
}


# ---------------------------------------------------------------------------
# Correspondência (matching)
# ---------------------------------------------------------------------------

# Teste de razão de Lowe: um casamento só é aceito se a distância ao vizinho
# mais próximo for menor que RAZAO_LOWE vezes a distância ao segundo vizinho.
RAZAO_LOWE = 0.75


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

# Número máximo de casamentos desenhados nas figuras de correspondência,
# para não poluir a imagem quando há centenas de casamentos. A amostra usa
# a semente fixa (SEMENTE) para ser reprodutível.
MAX_CASAMENTOS_DESENHADOS = 80

CORES_DETECTORES = {
    "Harris": "#1f77b4",
    "SIFT": "#2ca02c",
    "ORB": "#d62728",
}

NOMES_DETECTORES = ["Harris", "SIFT", "ORB"]
NOMES_DESCRITORES = ["SIFT", "ORB"]
