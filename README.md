# Preditor de preço de carros usados — mercado espanhol

Checkpoint 4 — Data Science & Statistical Computing — FIAP 2026
Prof. Jones Egydio

## Integrantes do grupo

| Nome | RM |
|---|---|
| Leonardo Eiji Kina | 562784 |
| Nicholas Braga de Souza | 561733 |
| Tomé Rossi Giani | 562422 |
| Vitor Ramos de Farias | 561958 |

---

## Objetivo

Estimar o **preço de anúncio** de veículos usados no mercado espanhol a partir
de características observáveis (idade, quilometragem, potência, câmbio,
combustível e marca), usando regressão linear.

O projeto tem finalidade tanto preditiva quanto explicativa: os coeficientes
são interpretados em euros para quantificar a contribuição de cada
característica.

**Pergunta de pesquisa.** Em que medida idade, quilometragem, potência e tipo
de câmbio explicam o preço anunciado de um carro usado no mercado espanhol, e
a relação entre idade e preço é linear?

---

## Origem dos dados

| | |
|---|---|
| **Base** | Coches de segunda mano Milanuncios |
| **Autores** | Iván Maseda Zurdo e Lucas Rey Pitaluga |
| **Repositório** | Zenodo — DOI [10.5281/zenodo.4674757](https://doi.org/10.5281/zenodo.4674757) |
| **Licença** | CC BY-NC-SA 4.0 |
| **Coleta** | Anúncios do portal milanuncios.com, extraídos em 09/04/2021 |
| **Unidade de observação** | Um anúncio de veículo |
| **Dimensões originais** | 500 linhas × 21 colunas |
| **Dimensões após tratamento** | 484 linhas × 14 colunas |
| **Arquivo** | `coches_milanuncios_09_04_2021.csv` (530,1 kB) |
| **MD5** | `8ab386b96887fd4738241f84d320a5c6` |

A base foi produzida no contexto da disciplina *Tipología y ciclo de vida de
los datos*, do Máster en Ciencia de Datos da Universitat Oberta de Catalunya.

O MD5 do arquivo utilizado foi conferido contra o publicado no Zenodo,
confirmando que se trata da versão com DOI.

### Nota sobre a documentação da fonte

Duas divergências foram identificadas entre a documentação e o conteúdo real,
e estão tratadas no relatório:

1. **Data de referência ambígua.** A descrição no Zenodo informa extração em
   09/04/2020, mas o arquivo se chama `_09_04_2021.csv` e a publicação é de
   09/04/2021. Como a base contém 16 veículos de ano 2021 — impossíveis em uma
   extração de 2020 — adotou-se **2021** como ano de referência para o cálculo
   da idade.
2. **Categorias de combustível incompletas.** A documentação declara apenas
   `gasolina` e `diesel`; a base contém 57 registros `híbrido`.

---

## Estrutura dos arquivos

```
.
├── app.py                  Aplicação Streamlit
├── treinar_modelo.py       Treina o modelo final e gera modelo/modelo.pkl
├── wrangling.py            Limpeza determinística da base original
├── eda.py                  Gera as figuras da análise exploratória
├── diagnostico.py          Diagnóstico do modelo final (resíduos, VIF)
├── modelagem.py            Os quatro modelos e a comparação
├── notebook.ipynb          Análise completa
├── requirements.txt        Dependências com versões fixadas
├── README.md
├── dados/
│   ├── coches.csv          Base original (Zenodo)
│   └── coches_tratado.csv  Base após limpeza determinística
├── modelo/
│   ├── modelo.pkl          Pipeline serializado + metadados
│   └── metricas.json       Métricas e intervalos, em formato legível
└── figuras/                Figuras geradas pela EDA e pelo diagnóstico Figuras geradas pela EDA e pelo diagnóstico
```

---

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

As versões estão fixadas no `requirements.txt` por um motivo específico: o
pickle do scikit-learn não é compatível entre versões diferentes da
biblioteca. Um modelo serializado com a 1.8.0 pode falhar ao ser carregado por
outra versão.

---

## Reprodução completa

A ordem abaixo reproduz o projeto do arquivo original até a aplicação.

```bash
# 1. Obter a base original (ou baixar manualmente do Zenodo)
mkdir -p dados
curl -sL "https://raw.githubusercontent.com/lreyp/Scraping-Milanuncios-Coches/main/csv/coches_milanuncios_09_04_2021.csv" \
     -o dados/coches.csv

# 2. Conferir integridade (deve resultar em 8ab386b96887fd4738241f84d320a5c6)
md5sum dados/coches.csv

# 3. Limpeza determinística -> dados/coches_tratado.csv
python wrangling.py

# 4. Análise exploratória -> figuras/
python eda.py

# 5. Comparação dos quatro modelos
python modelagem.py

# 6. Diagnóstico do modelo final
python diagnostico.py

# 7. Treinar e serializar o modelo -> modelo/modelo.pkl
python treinar_modelo.py

# 8. Executar a aplicação
streamlit run app.py
```

O notebook `notebook.ipynb` contém a análise completa com as interpretações
escritas e pode ser executado de forma independente.

---

## Execução da aplicação

```bash
streamlit run app.py
```

A aplicação abre em `http://localhost:8501` e contém quatro abas:

- **Fazer uma previsão** — formulário com as variáveis explicativas, resultado
  em euros, avisos de extrapolação e indicação de confiabilidade por faixa de
  preço.
- **Base de dados** — amostra e estatísticas descritivas.
- **Exploração** — dois gráficos exploratórios com interpretação.
- **Desempenho detalhado** — métricas, real vs previsto, resíduos e erro
  relativo por faixa.

### Consistência entre notebook e aplicação

A entrada do usuário passa pelo **mesmo objeto `Pipeline`** ajustado no
treinamento. Imputação, padronização, codificação one-hot e expansão
polinomial são os transformadores serializados no `modelo.pkl` — não há
implementação paralela na aplicação.

---

## Modelo final

**Regressão linear com expansão polinomial de grau 2 em `idade`**, domínio
restrito a veículos de até 20 anos.

Divisão treino/teste 70/30 com `random_state=42`. Todas as transformações
aprendidas são ajustadas exclusivamente no conjunto de treino.

### Desempenho no conjunto de teste (144 anúncios)

| Métrica | Valor |
|---|---:|
| MAE | 3.486 € |
| RMSE | 5.912 € |
| R² | 0,749 |
| Erro absoluto mediano | 2.300 € |

### Comparação entre os modelos

| Modelo | MAE (€) | RMSE (€) | R² |
|---|---:|---:|---:|
| Referência (média) | 8.052 | 11.805 | −0,000 |
| Linear simples (`cv`) | 6.227 | 8.803 | 0,444 |
| Linear múltipla | 3.464 | 5.974 | 0,744 |
| Polinomial (grau 2 em idade) | 3.486 | 5.912 | 0,749 |

O ganho do modelo polinomial sobre o múltiplo é pequeno e não uniforme:
validação cruzada 5-fold indica MAE de 3.272 € contra 3.361 €, uma vantagem de
89 € com desvio-padrão próximo de 390 €. A justificativa para adotá-lo é a
evidência de não linearidade documentada na exploração, não a diferença de
métricas.

---

## Limitações conhecidas

### Da base

1. **Tamanho reduzido.** 500 registros originais, 484 após tratamento. Limita a
   estabilidade dos coeficientes e obrigou a agrupar 19 marcas raras.
2. **Recorte único.** Uma só data de coleta (09/04/2021). Não capta variação
   temporal de preços.
3. **Mercado espanhol.** Preços em euros, condições do mercado local. Não
   transponível para outro país sem nova coleta.
4. **Vendedores quase todos profissionais.** 497 de 500 anúncios. As conclusões
   refletem precificação de concessionária, não venda entre particulares.
5. **Variáveis ausentes relevantes.** Estado de conservação, histórico de
   manutenção, número de proprietários e urgência do vendedor não constam da
   base — e são determinantes de preço, sobretudo em veículos antigos.
6. **Preço pedido, não de transação.** O modelo estima o que o vendedor pede,
   não o valor efetivamente negociado.

### Do modelo

7. **Heteroscedasticidade.** A variância dos erros cresce com o preço
   (Spearman entre |resíduo| e ajustado: ρ = 0,22; p = 0,008).
8. **Resíduos não normais.** Shapiro-Wilk p ≈ 4 × 10⁻¹⁴; curtose 14,4.
   Intervalos de confiança baseados em normalidade não são confiáveis.
9. **Erro relativo elevado em veículos baratos.** Cerca de 52% do preço na
   faixa de 1.000 a 6.000 €, contra 14% acima de 19.000 €.
10. **Previsões negativas possíveis.** A regressão linear não impõe restrição
    de positividade. Combinações de idade e quilometragem elevadas produzem
    valores negativos (por exemplo, 18 anos e 250.000 km resultam em −1.753 €).
    A aplicação detecta o caso, limita a estimativa ao menor preço observado e
    informa o usuário de que o resultado não é confiável.
11. **Subestimação de veículos caros e de alta potência.** O efeito de `cv` é
    tratado como linear, o que não descreve bem os segmentos esportivo e
    premium.
12. **Domínio restrito a 20 anos.** Veículos mais antigos foram excluídos do
    modelo por serem precificados por raridade, não por depreciação.

### Metodológicas

13. **Associação, não causalidade.** O desenho é observacional e transversal,
    sem atribuição aleatória e com variáveis omitidas relevantes. Nenhum
    coeficiente deve ser lido como efeito de intervenção.
14. **Independência dos erros não verificável.** Anúncios da mesma
    concessionária podem compartilhar política de preços. A coluna `vendedor`
    foi descartada da modelagem, o que impede avaliar esse efeito.

---

## Uso de inteligência artificial

Ferramentas de IA foram utilizadas como apoio no desenvolvimento. Todas as
decisões de limpeza, modelagem e interpretação foram revisadas pelo grupo, e
os integrantes são responsáveis por todo o conteúdo entregue.

---

## Licença dos dados

A base é distribuída sob **CC BY-NC-SA 4.0**, que exige atribuição, veda uso
comercial e requer compartilhamento sob a mesma licença.

> Maseda Zurdo, I., & Rey Pitaluga, L. (2021). *Coches de segunda mano
> Milanuncios* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4674757

Observação: a descrição do registro no Zenodo indica CC BY-NC-SA 4.0, enquanto
o campo de metadados da plataforma registra CC BY 4.0. Adotou-se a licença mais
restritiva das duas.
# DS-SC-CP-4
