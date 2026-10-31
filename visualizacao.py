"""Geração de figuras: verificação da homografia, pontos detectados, gráficos e casamentos.

Usa o backend "Agg" do matplotlib (sem janela), já que o objetivo aqui é
salvar arquivos PNG em `resultados/`, não exibir nada interativamente.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def _bgr_para_rgb(imagem: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)


def figura_verificacao_homografia(
    img1_bgr: np.ndarray, img2_bgr: np.ndarray, homografia: np.ndarray, roi2: np.ndarray, caminho_saida
) -> None:
    """Sobrepõe a imagem 1 (deformada por H) sobre a imagem 2, para checar a homografia a olho.

    Se a homografia (calculada a partir dos pontos marcados manualmente)
    estiver correta, as estruturas do plano principal (janelas, bordas,
    mosaicos) devem coincidir nas duas camadas sobrepostas.
    """

    altura, largura = img2_bgr.shape[:2]
    img1_deformada = cv2.warpPerspective(img1_bgr, homografia, (largura, altura))

    sobreposta = cv2.addWeighted(img2_bgr, 0.5, img1_deformada, 0.5, 0)

    fig, eixos = plt.subplots(1, 3, figsize=(15, 5))
    eixos[0].imshow(_bgr_para_rgb(img2_bgr))
    eixos[0].set_title("Imagem 2 (original)")
    eixos[1].imshow(_bgr_para_rgb(img1_deformada))
    eixos[1].set_title("Imagem 1 deformada por H")
    eixos[2].imshow(_bgr_para_rgb(sobreposta))
    poligono_fechado = np.vstack([roi2, roi2[:1]])
    eixos[2].plot(poligono_fechado[:, 0], poligono_fechado[:, 1], "-y", linewidth=1.5)
    eixos[2].set_title("Sobreposição (ROI projetada em amarelo)")

    for eixo in eixos:
        eixo.axis("off")

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def figura_pontos(
    nome_detector: str,
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    caminho_saida,
) -> None:
    """Desenha os keypoints de um detector nas duas imagens do par, lado a lado."""

    desenho1 = cv2.drawKeypoints(
        img1_bgr, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    desenho2 = cv2.drawKeypoints(
        img2_bgr, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    fig, eixos = plt.subplots(1, 2, figsize=(12, 6))
    eixos[0].imshow(_bgr_para_rgb(desenho1))
    eixos[0].set_title(f"{nome_detector} — imagem 1 ({len(keypoints1)} pontos)")
    eixos[1].imshow(_bgr_para_rgb(desenho2))
    eixos[1].set_title(f"{nome_detector} — imagem 2 ({len(keypoints2)} pontos)")
    for eixo in eixos:
        eixo.axis("off")

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def graficos_variacao_parametro(
    df_deteccao: pd.DataFrame, nome_detector: str, nome_parametro: str, colunas_fixas: dict, caminho_saida
) -> None:
    """Gráficos de nº de pontos e repetibilidade vs. um parâmetro, com uma linha por par.

    `colunas_fixas` trava os demais parâmetros do detector no valor padrão,
    para isolar o efeito de `nome_parametro` (os outros parâmetros variam
    em testes separados, ver config.VARIACOES_*).
    """

    subconjunto = df_deteccao[df_deteccao["detector"] == nome_detector].copy()
    for coluna, valor in colunas_fixas.items():
        subconjunto = subconjunto[subconjunto[coluna] == valor]

    if subconjunto.empty:
        return

    subconjunto = subconjunto.sort_values(nome_parametro)

    fig, (eixo_pontos, eixo_repet) = plt.subplots(1, 2, figsize=(12, 5))

    for nome_par, grupo in subconjunto.groupby("par"):
        grupo = grupo.sort_values(nome_parametro)
        eixo_pontos.plot(grupo[nome_parametro], grupo["n_pontos_img1"], marker="o", label=nome_par)
        eixo_repet.plot(grupo[nome_parametro], grupo["repetibilidade"], marker="o", label=nome_par)

    eixo_pontos.set_xlabel(nome_parametro)
    eixo_pontos.set_ylabel("nº de pontos detectados (imagem 1)")
    eixo_pontos.set_title(f"{nome_detector}: nº de pontos vs. {nome_parametro}")
    eixo_pontos.legend(fontsize=8)

    eixo_repet.set_xlabel(nome_parametro)
    eixo_repet.set_ylabel("repetibilidade")
    eixo_repet.set_title(f"{nome_detector}: repetibilidade vs. {nome_parametro}")
    eixo_repet.set_ylim(0, 1.05)
    eixo_repet.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def _heatmap(matriz: pd.DataFrame, titulo: str, rotulo_cor: str, caminho_saida, formato: str = "{:.2f}") -> None:
    fig, eixo = plt.subplots(figsize=(1.2 * len(matriz.columns) + 2, 1.2 * len(matriz.index) + 2))
    mapa = eixo.imshow(matriz.values, cmap="viridis", vmin=0, vmax=1)

    eixo.set_xticks(range(len(matriz.columns)))
    eixo.set_xticklabels(matriz.columns)
    eixo.set_yticks(range(len(matriz.index)))
    eixo.set_yticklabels(matriz.index)
    eixo.set_title(titulo)

    for i in range(len(matriz.index)):
        for j in range(len(matriz.columns)):
            valor = matriz.values[i, j]
            cor_texto = "white" if valor < 0.6 else "black"
            eixo.text(j, i, formato.format(valor), ha="center", va="center", color=cor_texto)

    fig.colorbar(mapa, ax=eixo, label=rotulo_cor)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def heatmap_sobreposicao(matriz: pd.DataFrame, nome_par: str, caminho_saida) -> None:
    """Heatmap detector(linha) x detector(coluna): % de pontos da linha com vizinho na coluna."""

    _heatmap(matriz, f"Sobreposição entre detectores — {nome_par}", "fração de sobreposição", caminho_saida)


def heatmap_taxa_acerto(matriz: pd.DataFrame, nome_par: str, caminho_saida) -> None:
    """Heatmap detector(linha) x descritor(coluna): taxa de acerto do casamento."""

    _heatmap(matriz, f"Taxa de acerto detector x descritor — {nome_par}", "taxa de acerto", caminho_saida)


def figura_casamentos(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    casamentos: list[cv2.DMatch],
    corretos: np.ndarray,
    titulo: str,
    caminho_saida,
    semente: int = config.SEMENTE,
) -> None:
    """Desenha os casamentos: verde = correto (segundo a homografia de referência), vermelho = errado.

    Quando há muitos casamentos, desenha uma amostra aleatória (com
    semente fixa) de até `config.MAX_CASAMENTOS_DESENHADOS`, metade
    corretos e metade errados quando possível, para a figura não virar um
    emaranhado ilegível de linhas.
    """

    rng = np.random.default_rng(semente)

    indices_corretos = np.nonzero(corretos)[0]
    indices_errados = np.nonzero(~corretos)[0]

    limite = config.MAX_CASAMENTOS_DESENHADOS
    limite_por_classe = max(1, limite // 2)

    escolhidos = []
    for indices in (indices_corretos, indices_errados):
        if len(indices) > limite_por_classe:
            escolhidos.append(rng.choice(indices, size=limite_por_classe, replace=False))
        else:
            escolhidos.append(indices)
    indices_amostrados = np.concatenate(escolhidos) if escolhidos else np.array([], dtype=int)

    casamentos_amostrados = [casamentos[i] for i in indices_amostrados]
    # Cores em BGR (a tela é desenhada com OpenCV): verde = correto, vermelho = errado.
    cores = [(0, 200, 0) if corretos[i] else (0, 0, 220) for i in indices_amostrados]

    altura = max(img1_bgr.shape[0], img2_bgr.shape[0])
    largura_total = img1_bgr.shape[1] + img2_bgr.shape[1]
    tela = np.full((altura, largura_total, 3), 255, dtype=np.uint8)
    tela[: img1_bgr.shape[0], : img1_bgr.shape[1]] = img1_bgr
    deslocamento_x = img1_bgr.shape[1]
    tela[: img2_bgr.shape[0], deslocamento_x : deslocamento_x + img2_bgr.shape[1]] = img2_bgr

    for match, cor in zip(casamentos_amostrados, cores):
        p1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
        p2_original = keypoints2[match.trainIdx].pt
        p2 = (int(round(p2_original[0])) + deslocamento_x, int(round(p2_original[1])))
        cv2.line(tela, p1, p2, cor, 1, lineType=cv2.LINE_AA)
        cv2.circle(tela, p1, 3, cor, -1, lineType=cv2.LINE_AA)
        cv2.circle(tela, p2, 3, cor, -1, lineType=cv2.LINE_AA)

    n_corretos = int(np.sum(corretos))
    n_total = len(corretos)
    fig, eixo = plt.subplots(figsize=(14, 7))
    eixo.imshow(_bgr_para_rgb(tela))
    eixo.set_title(f"{titulo} — {n_corretos}/{n_total} corretos (amostra de {len(indices_amostrados)} desenhada)")
    eixo.axis("off")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
