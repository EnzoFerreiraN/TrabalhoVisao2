"""Ferramenta de anotação manual: ROI do plano principal + pontos correspondentes.

Por quê uma ferramenta manual? A taxa de acerto e a repetibilidade medem o
quão bons são os detectores/descritores -- então a homografia usada como
"gabarito" não pode vir do próprio RANSAC dos casamentos SIFT/ORB (isso
seria circular: usar o algoritmo para avaliar o algoritmo). A homografia
de referência vem, em vez disso, de pontos marcados à mão.

Uso (linha de comando):
    python anotacao.py                  # anota todos os pares principais sem anotação ainda
    python anotacao.py --par NOME       # anota só um par específico
    python anotacao.py --extras         # inclui também os pares extras (config.PARES_EXTRAS)
    python anotacao.py --refazer        # anota de novo mesmo que já exista JSON salvo

Etapas, por par:
    (a) Marque o polígono da ROI do plano principal NA IMAGEM 1 (clique
        esquerdo adiciona vértice, 'u' desfaz o último, Enter fecha o
        polígono com no mínimo 3 vértices).
    (b) Marque de 8 a 12 pontos correspondentes bem distribuídos sobre o
        plano, alternando clique na imagem 1 e na imagem 2 (mesma janela,
        lado a lado). 'u' desfaz o último clique, Enter termina quando
        houver pontos suficientes, Esc cancela a anotação do par.

Ao terminar os cliques, a homografia é calculada e o erro de reprojeção de
cada ponto é impresso no terminal, para o usuário validar a qualidade dos
cliques antes de salvar.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

import config
from homografia import calcular_homografia, caminho_anotacao, erro_reprojecao

JANELA = "Anotacao"


def _ler_imagem_original(caminho) -> np.ndarray:
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho}")
    return imagem


def _anotar_roi(imagem: np.ndarray) -> np.ndarray:
    """Coleta os vértices do polígono da ROI na imagem fornecida."""

    pontos: list[tuple[float, float]] = []

    def callback(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            pontos.append((float(x), float(y)))

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(JANELA, callback)

    print(
        "\n[ROI] Clique com o botão esquerdo para marcar os vértices do polígono "
        "do plano principal (na ordem do contorno). 'u' desfaz, Enter confirma "
        "(mínimo 3 pontos), Esc cancela."
    )

    while True:
        desenho = imagem.copy()
        for i, p in enumerate(pontos):
            cv2.circle(desenho, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(desenho, (int(pontos[i - 1][0]), int(pontos[i - 1][1])), (int(p[0]), int(p[1])), (0, 255, 255), 2)
        if len(pontos) >= 3:
            cv2.line(desenho, (int(pontos[-1][0]), int(pontos[-1][1])), (int(pontos[0][0]), int(pontos[0][1])), (0, 255, 255), 1)

        cv2.putText(
            desenho,
            f"ROI: {len(pontos)} pontos | Enter=ok (min 3)  u=desfazer  Esc=cancelar",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow(JANELA, desenho)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla in (13, 10):  # Enter
            if len(pontos) >= 3:
                break
        elif tecla in (ord("u"), ord("U")):
            if pontos:
                pontos.pop()
        elif tecla == 27:  # Esc
            cv2.destroyWindow(JANELA)
            raise KeyboardInterrupt("Anotação da ROI cancelada pelo usuário.")

    return np.array(pontos, dtype=np.float64)


def _anotar_correspondencias(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coleta pontos correspondentes clicando alternadamente na imagem 1 e na imagem 2."""

    altura = max(img1.shape[0], img2.shape[0])
    deslocamento_x = img1.shape[1]
    largura_total = img1.shape[1] + img2.shape[1]

    pontos1: list[tuple[float, float]] = []
    pontos2: list[tuple[float, float]] = []
    # Cada clique alterna entre imagem 1 e imagem 2; `proximo_lado` indica
    # qual das duas listas recebe o próximo clique válido.
    proximo_lado = {"valor": 1}

    def callback(evento, x, y, flags, param):
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= deslocamento_x:
            lado, x_rel = 2, x - deslocamento_x
        else:
            lado, x_rel = 1, x

        if lado != proximo_lado["valor"]:
            return  # ignora clique fora de ordem; usuário deve alternar 1, 2, 1, 2, ...

        if lado == 1:
            pontos1.append((float(x_rel), float(y)))
            proximo_lado["valor"] = 2
        else:
            pontos2.append((float(x_rel), float(y)))
            proximo_lado["valor"] = 1

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(JANELA, callback)

    print(
        "\n[Correspondências] Clique o MESMO ponto físico primeiro na imagem da "
        "ESQUERDA, depois na imagem da DIREITA, alternando sempre. Marque de "
        f"{config.MIN_PONTOS_ANOTADOS} a {config.MAX_PONTOS_ANOTADOS} pontos bem "
        "espalhados pelo plano. 'u' desfaz o último clique, Enter termina, Esc cancela."
    )

    while True:
        tela = np.full((altura, largura_total, 3), 50, dtype=np.uint8)
        tela[: img1.shape[0], : img1.shape[1]] = img1
        tela[: img2.shape[0], deslocamento_x : deslocamento_x + img2.shape[1]] = img2

        for i, p in enumerate(pontos1):
            centro = (int(p[0]), int(p[1]))
            cv2.circle(tela, centro, 6, (0, 255, 0), -1)
            cv2.putText(tela, str(i + 1), (centro[0] + 8, centro[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for i, p in enumerate(pontos2):
            centro = (int(p[0]) + deslocamento_x, int(p[1]))
            cv2.circle(tela, centro, 6, (0, 255, 0), -1)
            cv2.putText(tela, str(i + 1), (centro[0] + 8, centro[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        lado_txt = "ESQUERDA" if proximo_lado["valor"] == 1 else "DIREITA"
        cv2.putText(
            tela,
            f"Pontos: {len(pontos2)} pares completos | próximo clique: {lado_txt} | "
            f"Enter=ok (min {config.MIN_PONTOS_ANOTADOS})  u=desfazer  Esc=cancelar",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow(JANELA, tela)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla in (13, 10):
            if len(pontos2) >= config.MIN_PONTOS_ANOTADOS and len(pontos1) == len(pontos2):
                break
        elif tecla in (ord("u"), ord("U")):
            if proximo_lado["valor"] == 2 and pontos1:
                pontos1.pop()
                proximo_lado["valor"] = 1
            elif proximo_lado["valor"] == 1 and pontos2:
                pontos2.pop()
                proximo_lado["valor"] = 2
        elif tecla == 27:
            cv2.destroyWindow(JANELA)
            raise KeyboardInterrupt("Anotação de correspondências cancelada pelo usuário.")

        if len(pontos1) >= config.MAX_PONTOS_ANOTADOS and len(pontos2) >= config.MAX_PONTOS_ANOTADOS:
            break

    return np.array(pontos1, dtype=np.float64), np.array(pontos2, dtype=np.float64)


def anotar_par(par: config.Par, pasta=config.PASTA_ANOTACOES, refazer: bool = False) -> None:
    """Executa a anotação interativa completa de um par e salva o JSON resultante."""

    destino = caminho_anotacao(par, pasta)
    if destino.exists() and not refazer:
        print(f"[{par.nome}] já anotado ({destino}); use --refazer para anotar de novo.")
        return

    print(f"\n=== Anotando par '{par.nome}' ({par.imagem1} / {par.imagem2}) ===")
    img1 = _ler_imagem_original(config.PASTA_IMAGENS / par.imagem1)
    img2 = _ler_imagem_original(config.PASTA_IMAGENS / par.imagem2)

    while True:
        try:
            roi1 = _anotar_roi(img1)
            pontos1, pontos2 = _anotar_correspondencias(img1, img2)
        except KeyboardInterrupt as erro:
            print(f"[{par.nome}] {erro}")
            cv2.destroyAllWindows()
            return

        H = calcular_homografia(pontos1, pontos2)
        erros = erro_reprojecao(H, pontos1, pontos2)

        print(f"\n[{par.nome}] Erros de reprojeção dos pontos marcados (pixels, imagem original):")
        for i, erro in enumerate(erros, start=1):
            print(f"   ponto {i:2d}: {erro:6.2f} px")
        print(f"   média: {erros.mean():.2f} px | máximo: {erros.max():.2f} px")

        resposta = input("\nSalvar esta anotação? [S]im / [r]efazer / [c]ancelar: ").strip().lower()
        if resposta in ("", "s", "sim"):
            break
        if resposta in ("c", "cancelar"):
            cv2.destroyAllWindows()
            print(f"[{par.nome}] anotação descartada.")
            return
        # qualquer outra resposta (ex.: "r") refaz os cliques

    cv2.destroyAllWindows()

    dados = {
        "imagem1": par.imagem1,
        "imagem2": par.imagem2,
        "tamanho_original1": [img1.shape[1], img1.shape[0]],
        "tamanho_original2": [img2.shape[1], img2.shape[0]],
        "roi_imagem1": roi1.tolist(),
        "pontos_imagem1": pontos1.tolist(),
        "pontos_imagem2": pontos2.tolist(),
        "erro_medio_px": float(erros.mean()),
    }

    pasta.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)

    print(f"[{par.nome}] anotação salva em {destino} (erro médio: {erros.mean():.2f} px).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferramenta de anotação manual (ROI + pontos correspondentes).")
    parser.add_argument("--par", type=str, default=None, help="Nome de um único par a anotar.")
    parser.add_argument("--extras", action="store_true", help="Inclui também os pares extras de config.PARES_EXTRAS.")
    parser.add_argument("--refazer", action="store_true", help="Anota de novo mesmo que já exista JSON salvo.")
    args = parser.parse_args()

    pares = config.pares_ativos(incluir_extras=args.extras)

    if args.par:
        pares = [p for p in pares if p.nome == args.par]
        if not pares:
            print(f"Par '{args.par}' não encontrado entre os pares configurados.")
            sys.exit(1)

    for par in pares:
        anotar_par(par, refazer=args.refazer)


if __name__ == "__main__":
    main()
