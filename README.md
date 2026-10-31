# Atividade Prática 02 

Comparação de detectores de pontos de interesse (Harris, SIFT, ORB) e de
descritores (SIFT, ORB) em pares de imagens de cenas planas.

## Estrutura do código

O código é dividido em módulos pequenos, cada um com uma responsabilidade:

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Todos os parâmetros e caminhos do projeto (pares de imagens, faixas de parâmetros de cada detector, ε de erro em pixels, razão de Lowe, tamanho de redimensionamento, semente aleatória, etc). Se quiser mudar algum número do experimento, é aqui. |
| `imagens.py` | Carrega imagens (já com a orientação EXIF corrigida pelo próprio OpenCV), redimensiona para o lado maior = 1000 px, converte para cinza, cria a máscara da ROI do plano principal. |
| `anotacao.py` | Ferramenta interativa (cliques de mouse) para marcar a ROI do plano e os pontos correspondentes entre as duas imagens de cada par. Também é um script de linha de comando. |
| `homografia.py` | Calcula a homografia de referência (ground truth) a partir dos pontos marcados manualmente, e as máscaras de ROI na escala de trabalho. |
| `detectores.py` | Uma função por detector (`detectar_harris`, `detectar_sift`, `detectar_orb`), todas com a mesma assinatura: imagem + parâmetros + máscara → lista de `cv2.KeyPoint`. |
| `descritores.py` | Computa descritores SIFT e ORB para qualquer lista de keypoints, mesmo as vindas de outro detector. |
| `correspondencia.py` | Casamento de descritores (BFMatcher + teste de razão de Lowe) e verificação de acerto via homografia de referência. |
| `metricas.py` | Repetibilidade, sobreposição entre detectores e taxa de acerto. |
| `visualizacao.py` | Todas as figuras: verificação da homografia, pontos desenhados, gráficos de variação de parâmetro, heatmaps e casamentos corretos/errados. |
| `main.py` | Roda as questões 1 e 2 para todos os pares configurados e salva os resultados. |

## Instalação

```bash
pip install -r requirements.txt
```

Requer Python 3.10+ e `opencv-python`, `numpy`, `matplotlib`, `pandas`.

## Como rodar

### 1. Anotar os pares (passo manual, feito uma vez por par)

A homografia usada como referência (gabarito) para medir repetibilidade e
taxa de acerto **não** pode vir do RANSAC dos próprios casamentos
SIFT/ORB — isso seria circular (usar o algoritmo para avaliar o próprio
algoritmo). Por isso, antes de rodar `main.py`, é preciso marcar à mão:

```bash
python anotacao.py
```

Isso abre, para cada par ainda sem anotação, duas janelas interativas, nesta ordem:

1. **ROI do plano principal** — clique com o botão esquerdo para marcar os
   vértices do polígono que delimita o plano principal da cena **na
   imagem 1** (ex.: só a fachada de frente, excluindo a fachada lateral,
   o céu, a água, a cabeça no rodapé, prédios vizinhos em outro plano,
   etc). `u` desfaz o último vértice, Enter fecha o polígono (mínimo 3
   vértices), Esc cancela a anotação do par.
2. **Pontos correspondentes** — as duas imagens aparecem lado a lado na
   mesma janela. Clique o mesmo ponto físico primeiro na imagem da
   esquerda, depois na da direita, alternando sempre, até marcar de 8 a
   12 pares bem distribuídos pelo plano. `u` desfaz o último clique,
   Enter termina, Esc cancela.

Ao final de cada par, o terminal imprime o erro de reprojeção (em pixels)
de cada ponto marcado e a média. **Valores altos (vários pixels) indicam
um clique ruim** — nesse caso, responda `r` quando perguntado para refazer
os cliques daquele par, em vez de salvar. Um erro médio de 1-2 px é um bom
sinal de anotação consistente.

As anotações são salvas em `anotacoes/<nome_do_par>.json`. Se o arquivo já
existir, o par é pulado (use `--refazer` para anotar de novo).

Opções de `anotacao.py`:

```bash
python anotacao.py --par dsc07631_dsc07632   # anota só esse par
python anotacao.py --extras                  # inclui também os pares extras (frames mais afastados)
python anotacao.py --refazer                 # reanota mesmo que já exista JSON salvo
```

### 2. Rodar os experimentos

```bash
python main.py
```

Isso processa os 3 pares principais (`dsc07631/dsc07632`,
`dsc07654/dsc07655` e o par próprio de Veneza), rodando a questão 1
(detectores) e a questão 2 (descritores + matching) para cada um.

Opções de `main.py`:

```bash
python main.py --extras        # inclui também dsc07631/dsc07635 e dsc07654/dsc07658
python main.py --sem-anotar    # pula pares sem anotação em vez de abortar com erro
```

## O que significa cada saída (pasta `resultados/`)

### `resultados/q1_deteccao.csv`

Uma linha por (par, detector, combinação de parâmetros). Colunas:
`n_pontos_img1`, `n_pontos_img2` (quantos pontos cada imagem produziu),
`tempo_seg` (tempo de detecção nas duas imagens), `repetibilidade`
(fração dos pontos da imagem 1 que, projetados pela homografia de
referência e caindo dentro da ROI da imagem 2, têm algum ponto da imagem
2 a até `config.EPSILON_PX` pixels de distância), `n_pontos_considerados`
e `n_pontos_repetidos` (numerador/denominador da repetibilidade), e as
colunas de parâmetros de cada detector (`k`, `blockSize`, `limiar` para
Harris; `contrastThreshold`, `edgeThreshold` para SIFT; `nfeatures`,
`fastThreshold` para ORB — ficam `NaN` quando não se aplicam à linha).

### `resultados/q1_sobreposicao.csv`

Uma linha por (par, imagem, detector A, detector B), com a configuração
padrão de cada detector: `fracao_sobreposicao` é a fração dos pontos do
detector A que têm algum ponto do detector B a até `config.EPSILON_PX`
pixels de distância, na mesma imagem (sem usar homografia — é uma
comparação espacial direta dentro de uma única foto).

### `resultados/q2_taxa_acerto.csv`

Uma linha por (par, detector, descritor). `n_keypoints_entrada_img1` é
quantos keypoints o detector produziu (configuração padrão);
`n_keypoints_descritos_img1/2` é quanto sobrou depois do `compute()` do
descritor (ele descarta keypoints muito perto da borda da imagem — ver
comentário em `descritores.py`). `n_casamentos` é quantos pares passaram
no teste de razão de Lowe; `n_corretos` é quantos desses casamentos têm
erro de reprojeção (via homografia de referência) menor que
`config.EPSILON_PX`; `taxa_acerto = n_corretos / n_casamentos`.

### `resultados/graficos/`

Um gráfico (duas figuras lado a lado: nº de pontos e repetibilidade) por
combinação (detector, parâmetro variado), com uma linha por par — os
demais parâmetros do detector ficam fixos no valor padrão
(`config.PADRAO_*`), para isolar o efeito do parâmetro em questão.

### `resultados/<nome_do_par>/`

Por par:

- `verificacao_homografia.png` — a imagem 1 deformada pela homografia de
  referência, sobreposta semitransparente à imagem 2, com o contorno da
  ROI projetada em amarelo. Serve para checar visualmente se a anotação
  manual ficou boa (as estruturas do plano devem coincidir nas duas
  camadas).
- `pontos_Harris.png`, `pontos_SIFT.png`, `pontos_ORB.png` — os keypoints
  da configuração padrão de cada detector, desenhados nas duas imagens.
- `sobreposicao.png` — heatmap detector × detector com a fração de
  sobreposição (média das duas imagens).
- `q2_heatmap.png` — heatmap detector × descritor com a taxa de acerto.
- `casamentos_<detector>_<descritor>.png` — os casamentos entre as duas
  imagens (verde = correto, vermelho = errado), para as 6 combinações da
  questão 2. Quando há muitos casamentos, desenha-se uma amostra (semente
  fixa) de até `config.MAX_CASAMENTOS_DESENHADOS`.

## Resumo no terminal

Ao final, `main.py` imprime, para cada par: o detector com maior
repetibilidade (configuração padrão) e a combinação detector+descritor
com maior taxa de acerto (desempate pelo número de casamentos corretos).

## Observações de projeto

- **Redimensionamento**: as imagens próprias (4000×3000) são reduzidas
  para o lado maior = 1000 px (`config.MAIOR_LADO`); os pares fornecidos
  pela disciplina (640×480) já são menores que isso e **não são
  ampliados** (`config.PERMITIR_AMPLIACAO = False`), porque ampliar não
  acrescenta informação real à imagem — apenas interpola pixels, o que
  mudaria artificialmente o comportamento dos detectores.
- **Harris não tem escala/orientação nativas**: para que SIFT e ORB
  consigam descrever os cantos do Harris na questão 2, cada canto vira um
  `cv2.KeyPoint` com tamanho fixo (`config.HARRIS_TAMANHO_KEYPOINT`).
- **Keypoints de um detector descritos por outro**: o campo `octave` de
  um `cv2.KeyPoint` tem significado específico para cada algoritmo (SIFT
  empacota oitava+camada da pirâmide de escala; ORB usa o nível da
  pirâmide de resolução). Por isso, antes de descrever um keypoint que
  não é nativo do descritor, `descritores.py` zera esse campo, mantendo
  posição, tamanho e ângulo — ver o comentário no topo do arquivo.
- **Pares extras** (`config.PARES_EXTRAS`, ligados com `--extras`): os
  pares `dsc07631/dsc07635` e `dsc07654/dsc07658` usam frames mais
  afastados dos mesmos pontos de vista que os pares principais `dsc`, que
  são frames consecutivos e quase idênticos (fazendo repetibilidade e
  taxa de acerto saturarem perto de 1.0). Úteis para comparar no
  relatório um caso com deformação de ponto de vista maior.
- **Reprodutibilidade**: `main.py` fixa a semente (`config.SEMENTE`) do
  NumPy e do OpenCV antes de qualquer etapa com aleatoriedade (amostragem
  de casamentos desenhados nas figuras).
