"""Orquestra as questões 1 e 2 da Atividade Prática 02 para todos os pares configurados.

Uso:
    python main.py                 # pares principais, exige anotação manual já feita
    python main.py --extras        # inclui também os pares extras (frames mais afastados)
    python main.py --sem-anotar    # pula pares sem anotação em vez de abortar

As saídas vão para `resultados/` (CSVs, gráficos e, por par, uma subpasta
com as figuras). Ao final, um resumo é impresso no terminal.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
import correspondencia
import descritores
import detectores
import homografia
import imagens
import metricas
import visualizacao


def _gerar_configuracoes(variacoes: dict[str, list]) -> list[dict]:
    """Produz a grade completa (produto cartesiano) de parâmetros de um detector."""

    chaves = list(variacoes.keys())
    listas_de_valores = [variacoes[chave] for chave in chaves]
    return [dict(zip(chaves, combinacao)) for combinacao in itertools.product(*listas_de_valores)]


def _detectar_com_tempo(nome_detector: str, cinza: np.ndarray, params: dict, mascara: np.ndarray):
    funcao = detectores.DETECTORES[nome_detector]
    inicio = time.perf_counter()
    keypoints = funcao(cinza, params, mascara)
    tempo = time.perf_counter() - inicio
    return keypoints, tempo


def processar_questao1(par_carregado: imagens.ParCarregado, referencia: homografia.ReferenciaPar, pasta_par: Path):
    """Roda a grade completa de parâmetros dos 3 detectores e mede repetibilidade.

    Devolve (linhas_deteccao, linhas_sobreposicao, keypoints_padrao), onde
    `keypoints_padrao` é um dicionário {detector: (kps_img1, kps_img2)} com
    os pontos obtidos na configuração padrão -- reaproveitados nas figuras,
    na sobreposição e na questão 2, para não detectar tudo de novo.
    """

    par = par_carregado.par
    linhas_deteccao = []
    keypoints_padrao: dict[str, tuple[list[cv2.KeyPoint], list[cv2.KeyPoint]]] = {}

    for nome_detector in config.NOMES_DETECTORES:
        variacoes = detectores.VARIACOES[nome_detector]
        configuracoes = _gerar_configuracoes(variacoes)

        for params in configuracoes:
            kps1, tempo1 = _detectar_com_tempo(nome_detector, par_carregado.img1_cinza, params, referencia.mascara1)
            kps2, tempo2 = _detectar_com_tempo(nome_detector, par_carregado.img2_cinza, params, referencia.mascara2)

            fracao_rep, n_considerados, n_repetidos = metricas.repetibilidade(
                kps1, kps2, referencia.homografia, referencia.mascara2
            )

            linha = {
                "par": par.nome,
                "detector": nome_detector,
                "n_pontos_img1": len(kps1),
                "n_pontos_img2": len(kps2),
                "tempo_seg": tempo1 + tempo2,
                "repetibilidade": fracao_rep,
                "n_pontos_considerados": n_considerados,
                "n_pontos_repetidos": n_repetidos,
            }
            linha.update(params)
            linhas_deteccao.append(linha)

        # Guarda os keypoints da configuração PADRÃO (reaproveitados abaixo).
        params_padrao = detectores.PARAMS_PADRAO[nome_detector]
        kps1_padrao, _ = _detectar_com_tempo(nome_detector, par_carregado.img1_cinza, params_padrao, referencia.mascara1)
        kps2_padrao, _ = _detectar_com_tempo(nome_detector, par_carregado.img2_cinza, params_padrao, referencia.mascara2)
        keypoints_padrao[nome_detector] = (kps1_padrao, kps2_padrao)

        visualizacao.figura_pontos(
            nome_detector,
            par_carregado.img1_bgr,
            par_carregado.img2_bgr,
            kps1_padrao,
            kps2_padrao,
            pasta_par / f"pontos_{nome_detector}.png",
        )

    # Sobreposição entre detectores (configuração padrão), em cada imagem.
    linhas_sobreposicao = []
    for imagem_idx, kps_por_detector in (
        (1, {d: keypoints_padrao[d][0] for d in config.NOMES_DETECTORES}),
        (2, {d: keypoints_padrao[d][1] for d in config.NOMES_DETECTORES}),
    ):
        for detector_a, detector_b in itertools.permutations(config.NOMES_DETECTORES, 2):
            fracao, n_pontos_a = metricas.sobreposicao(kps_por_detector[detector_a], kps_por_detector[detector_b])
            linhas_sobreposicao.append(
                {
                    "par": par.nome,
                    "imagem": imagem_idx,
                    "detector_a": detector_a,
                    "detector_b": detector_b,
                    "fracao_sobreposicao": fracao,
                    "n_pontos_a": n_pontos_a,
                }
            )

    # Heatmap de sobreposição (média das duas imagens, diagonal = 1.0).
    matriz = pd.DataFrame(1.0, index=config.NOMES_DETECTORES, columns=config.NOMES_DETECTORES)
    df_sobrep_par = pd.DataFrame(linhas_sobreposicao)
    for detector_a, detector_b in itertools.permutations(config.NOMES_DETECTORES, 2):
        media = df_sobrep_par[
            (df_sobrep_par["detector_a"] == detector_a) & (df_sobrep_par["detector_b"] == detector_b)
        ]["fracao_sobreposicao"].mean()
        matriz.loc[detector_a, detector_b] = media
    visualizacao.heatmap_sobreposicao(matriz, par.nome, pasta_par / "sobreposicao.png")

    return linhas_deteccao, linhas_sobreposicao, keypoints_padrao


def processar_questao2(
    par_carregado: imagens.ParCarregado,
    referencia: homografia.ReferenciaPar,
    keypoints_padrao: dict[str, tuple[list[cv2.KeyPoint], list[cv2.KeyPoint]]],
    pasta_par: Path,
):
    """Para cada combinação detector x descritor, descreve, casa e mede a taxa de acerto."""

    par = par_carregado.par
    linhas = []
    matriz_taxa = pd.DataFrame(
        np.nan, index=config.NOMES_DETECTORES, columns=config.NOMES_DESCRITORES
    )

    for nome_detector in config.NOMES_DETECTORES:
        kps1_entrada, kps2_entrada = keypoints_padrao[nome_detector]

        for nome_descritor in config.NOMES_DESCRITORES:
            funcao_descrever, norma = descritores.DESCRITORES[nome_descritor]

            inicio = time.perf_counter()
            kps1, desc1 = funcao_descrever(par_carregado.img1_cinza, kps1_entrada, nome_detector)
            kps2, desc2 = funcao_descrever(par_carregado.img2_cinza, kps2_entrada, nome_detector)
            casamentos = correspondencia.casar(desc1, desc2, norma)
            tempo = time.perf_counter() - inicio

            corretos = correspondencia.classificar_casamentos(kps1, kps2, casamentos, referencia.homografia)
            n_corretos = int(np.sum(corretos))
            taxa = metricas.taxa_acerto(n_corretos, len(casamentos))

            linhas.append(
                {
                    "par": par.nome,
                    "detector": nome_detector,
                    "descritor": nome_descritor,
                    "n_keypoints_entrada_img1": len(kps1_entrada),
                    "n_keypoints_descritos_img1": len(kps1),
                    "n_keypoints_descritos_img2": len(kps2),
                    "n_casamentos": len(casamentos),
                    "n_corretos": n_corretos,
                    "taxa_acerto": taxa,
                    "tempo_seg": tempo,
                }
            )
            matriz_taxa.loc[nome_detector, nome_descritor] = taxa

            visualizacao.figura_casamentos(
                par_carregado.img1_bgr,
                par_carregado.img2_bgr,
                kps1,
                kps2,
                casamentos,
                corretos,
                f"{par.nome}: {nome_detector}+{nome_descritor}",
                pasta_par / f"casamentos_{nome_detector}_{nome_descritor}.png",
            )

    visualizacao.heatmap_taxa_acerto(matriz_taxa, par.nome, pasta_par / "q2_heatmap.png")
    return linhas


def processar_par(par: config.Par, pasta_anotacoes: Path, pasta_resultados: Path, sem_anotar: bool):
    """Processa um par completo (Q1 + Q2); devolve None se a anotação estiver faltando e --sem-anotar."""

    if sem_anotar and not homografia.anotacao_existe(par, pasta_anotacoes):
        print(f"[AVISO] Par '{par.nome}' sem anotação em {pasta_anotacoes} -- pulado (--sem-anotar).")
        return None

    print(f"\n=== Processando par '{par.nome}' ===")
    par_carregado = imagens.carregar_par(par)
    referencia = homografia.obter_referencia(par_carregado, pasta_anotacoes)
    print(f"  Erro médio dos cliques manuais em relação à H de referência: {referencia.erro_medio_px:.2f} px")
    if config.REFINAR_HOMOGRAFIA_ECC:
        print(f"  Ajuste do refinamento ECC sobre a H manual (mediana na ROI): {referencia.ajuste_ecc_px:.2f} px")

    pasta_par = pasta_resultados / par.nome
    pasta_par.mkdir(parents=True, exist_ok=True)

    visualizacao.figura_verificacao_homografia(
        par_carregado.img1_bgr,
        par_carregado.img2_bgr,
        referencia.homografia,
        referencia.roi2,
        pasta_par / "verificacao_homografia.png",
    )

    linhas_deteccao, linhas_sobreposicao, keypoints_padrao = processar_questao1(par_carregado, referencia, pasta_par)
    linhas_q2 = processar_questao2(par_carregado, referencia, keypoints_padrao, pasta_par)

    # Resumo deste par: melhor detector por repetibilidade (configuração
    # padrão) e melhor combinação detector+descritor por taxa de acerto.
    df_deteccao_par = pd.DataFrame(linhas_deteccao)
    melhor_detector = None
    melhor_repetibilidade = -1.0
    for nome_detector in config.NOMES_DETECTORES:
        params_padrao = detectores.PARAMS_PADRAO[nome_detector]
        filtro = df_deteccao_par["detector"] == nome_detector
        for chave, valor in params_padrao.items():
            filtro &= df_deteccao_par[chave] == valor
        linha_padrao = df_deteccao_par[filtro].iloc[0]
        if linha_padrao["repetibilidade"] > melhor_repetibilidade:
            melhor_repetibilidade = linha_padrao["repetibilidade"]
            melhor_detector = nome_detector

    df_q2_par = pd.DataFrame(linhas_q2).sort_values(["taxa_acerto", "n_corretos"], ascending=False)
    melhor_combo = df_q2_par.iloc[0]

    resumo = {
        "par": par.nome,
        "melhor_detector_repetibilidade": melhor_detector,
        "repetibilidade": melhor_repetibilidade,
        "melhor_combo_detector": melhor_combo["detector"],
        "melhor_combo_descritor": melhor_combo["descritor"],
        "melhor_combo_taxa_acerto": melhor_combo["taxa_acerto"],
        "melhor_combo_n_corretos": int(melhor_combo["n_corretos"]),
    }

    return linhas_deteccao, linhas_sobreposicao, linhas_q2, resumo


def gerar_graficos_comparativos(df_deteccao: pd.DataFrame, pasta_graficos: Path) -> None:
    """Um gráfico por (detector, parâmetro variado), com uma linha por par -- os outros parâmetros ficam no padrão."""

    pasta_graficos.mkdir(parents=True, exist_ok=True)
    for nome_detector in config.NOMES_DETECTORES:
        params_padrao = detectores.PARAMS_PADRAO[nome_detector]
        for nome_parametro in detectores.VARIACOES[nome_detector]:
            outros_fixos = {chave: valor for chave, valor in params_padrao.items() if chave != nome_parametro}
            caminho = pasta_graficos / f"{nome_detector}_{nome_parametro}.png"
            visualizacao.graficos_variacao_parametro(df_deteccao, nome_detector, nome_parametro, outros_fixos, caminho)


def imprimir_resumo(resumos: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("RESUMO FINAL")
    print("=" * 78)
    for r in resumos:
        print(f"\nPar: {r['par']}")
        print(f"  Melhor detector por repetibilidade (config. padrão): {r['melhor_detector_repetibilidade']} "
              f"(repetibilidade = {r['repetibilidade']:.3f})")
        print(f"  Melhor combinação detector+descritor por taxa de acerto: "
              f"{r['melhor_combo_detector']}+{r['melhor_combo_descritor']} "
              f"(taxa = {r['melhor_combo_taxa_acerto']:.3f}, {r['melhor_combo_n_corretos']} casamentos corretos)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda as questões 1 e 2 da Atividade Prática 02.")
    parser.add_argument("--extras", action="store_true", help="Inclui também os pares extras (frames mais afastados).")
    parser.add_argument("--sem-anotar", action="store_true", help="Pula pares sem anotação, em vez de abortar.")
    parser.add_argument("--pasta-anotacoes", type=str, default=None, help="Sobrescreve a pasta de anotações (testes).")
    parser.add_argument("--pasta-resultados", type=str, default=None, help="Sobrescreve a pasta de resultados (testes).")
    args = parser.parse_args()

    # Semente fixa para toda e qualquer etapa com aleatoriedade (amostragem
    # de casamentos desenhados nas figuras), para resultados reproduzíveis.
    np.random.seed(config.SEMENTE)
    cv2.setRNGSeed(config.SEMENTE)

    pasta_anotacoes = Path(args.pasta_anotacoes) if args.pasta_anotacoes else config.PASTA_ANOTACOES
    pasta_resultados = Path(args.pasta_resultados) if args.pasta_resultados else config.PASTA_RESULTADOS
    pasta_graficos = pasta_resultados / "graficos"
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    pares = config.pares_ativos(incluir_extras=args.extras)

    todas_linhas_deteccao = []
    todas_linhas_sobreposicao = []
    todas_linhas_q2 = []
    resumos = []

    for par in pares:
        resultado = processar_par(par, pasta_anotacoes, pasta_resultados, args.sem_anotar)
        if resultado is None:
            continue
        linhas_deteccao, linhas_sobreposicao, linhas_q2, resumo = resultado
        todas_linhas_deteccao += linhas_deteccao
        todas_linhas_sobreposicao += linhas_sobreposicao
        todas_linhas_q2 += linhas_q2
        resumos.append(resumo)

    if not todas_linhas_deteccao:
        print("\nNenhum par foi processado (faltam anotações). Rode `python anotacao.py` primeiro.")
        return

    df_deteccao = pd.DataFrame(todas_linhas_deteccao)
    df_sobreposicao = pd.DataFrame(todas_linhas_sobreposicao)
    df_q2 = pd.DataFrame(todas_linhas_q2)

    df_deteccao.to_csv(pasta_resultados / "q1_deteccao.csv", index=False)
    df_sobreposicao.to_csv(pasta_resultados / "q1_sobreposicao.csv", index=False)
    df_q2.to_csv(pasta_resultados / "q2_taxa_acerto.csv", index=False)

    gerar_graficos_comparativos(df_deteccao, pasta_graficos)

    print(f"\nCSVs salvos em: {pasta_resultados}")
    print(f"Figuras por par em: {pasta_resultados}/<nome_do_par>/")
    print(f"Gráficos comparativos em: {pasta_graficos}")

    imprimir_resumo(resumos)


if __name__ == "__main__":
    main()
