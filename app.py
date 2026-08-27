"""
Preditor de preço de carros usados — mercado espanhol
Checkpoint 4, Data Science & Statistical Computing, FIAP 2026

Execução local:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Preço de carros usados — Espanha",
    page_icon="🚗",
    layout="wide",
)

AZUL, LARANJA, CINZA = "#2c6fbb", "#e07b39", "#7a7a7a"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

FONTE_URL = "https://doi.org/10.5281/zenodo.4674757"


# --------------------------------------------------------------------------
# Carregamento com cache
# --------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    """Cache de dados: o CSV não muda entre sessões."""
    return pd.read_csv("dados/coches_tratado.csv")


@st.cache_resource
def carregar_modelo():
    """Cache de recurso: o modelo é um objeto pesado, carregado uma vez.

    Estratégia de robustez: o pickle do scikit-learn é sensível à versão da
    biblioteca. Se o ambiente de publicação instalar uma versão diferente da
    usada no treinamento, o carregamento falha. Nesse caso a aplicação
    retreina o modelo a partir do mesmo script, garantindo que continue
    funcionando — e informa que isso ocorreu.
    """
    import joblib
    try:
        return joblib.load("modelo/modelo.pkl"), "arquivo"
    except Exception:
        from treinar_modelo import treinar
        return treinar(), "retreinado"


d = carregar_dados()
pacote, origem_modelo = carregar_modelo()

pipe = pacote["pipeline"]
met = pacote["metricas"]
intervalos = pacote["intervalos"]
opcoes = pacote["opcoes"]
NUM, CAT = pacote["colunas_num"], pacote["colunas_cat"]
IDADE_MAX = pacote["idade_maxima"]

# Base efetivamente usada pelo modelo (domínio restrito)
d_modelo = d[d["idade"] <= IDADE_MAX]


# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.title("🚗 Quanto vale este carro usado?")
st.markdown(
    "Estimativa do **preço de anúncio** de veículos usados no mercado espanhol, "
    "a partir de um modelo de regressão linear com termo polinomial."
)

if origem_modelo == "retreinado":
    st.warning(
        "O modelo serializado não pôde ser carregado (provável divergência de "
        "versão do scikit-learn) e foi retreinado nesta sessão. As previsões "
        "permanecem válidas."
    )

with st.expander("Sobre o problema, os dados e o modelo", expanded=False):
    st.markdown(f"""
**Pergunta de pesquisa.** Em que medida idade, quilometragem, potência e tipo
de câmbio explicam o preço anunciado de um carro usado, e a relação entre
idade e preço é linear?

**Variável resposta.** `precio` — preço pedido no anúncio, em euros.
Importante: é o preço **pedido**, não o preço de transação.

**Variáveis usadas na previsão.**

| Variável | Unidade | Papel |
|---|---|---|
| `idade` | anos | Preditor principal, com termo quadrático |
| `km` | quilômetros | Uso acumulado |
| `cv` | cavalos de vapor | Proxy de segmento |
| `puertas` | unidades | Tipo de carroceria |
| `transmision` | manual / automático | Diferencial de conforto |
| `combustible` | diesel / gasolina / híbrido | Segmento e custo de uso |
| `marca_agrupada` | 15 categorias | Posicionamento de marca |

**Fonte dos dados.** *Coches de segunda mano Milanuncios*, de Iván Maseda
Zurdo e Lucas Rey Pitaluga. Publicado no Zenodo sob DOI
[10.5281/zenodo.4674757]({FONTE_URL}), licença CC BY-NC-SA 4.0.
Anúncios coletados do portal milanuncios.com em 09/04/2021.

**Modelo.** Regressão linear com expansão polinomial de grau 2 em `idade`,
selecionada por evidência de não linearidade na depreciação. Domínio restrito
a veículos de até {IDADE_MAX} anos.

**Advertência.** Todos os resultados são associações observadas entre
anúncios. Não sustentam afirmações causais.
    """)

st.divider()

# --------------------------------------------------------------------------
# Métricas do modelo
# --------------------------------------------------------------------------
st.subheader("Desempenho do modelo no conjunto de teste")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{met['MAE']:,.0f} €",
          help="Erro absoluto médio: em média, a previsão se afasta esse tanto do preço real.")
c2.metric("RMSE", f"{met['RMSE']:,.0f} €",
          help="Raiz do erro quadrático médio. Penaliza mais os erros grandes, por isso supera o MAE.")
c3.metric("R²", f"{met['R2']:.3f}",
          help="Proporção da variação do preço explicada pelo modelo.")
c4.metric("Erro mediano", f"{met['erro_mediano']:,.0f} €",
          help="Metade das previsões erra menos que isso. Bem abaixo do MAE, "
               "porque poucos casos extremos elevam a média.")

st.caption(
    f"Avaliado em {met['n_teste']} anúncios não utilizados no treinamento "
    f"({met['n_treino']} anúncios de treino). Divisão 70/30, `random_state=42`."
)

st.divider()

# --------------------------------------------------------------------------
# Abas
# --------------------------------------------------------------------------
aba_prever, aba_dados, aba_explorar, aba_desempenho = st.tabs(
    ["Fazer uma previsão", "Base de dados", "Exploração", "Desempenho detalhado"]
)

# ==========================================================================
# ABA 1 — Previsão
# ==========================================================================
with aba_prever:
    st.subheader("Informe as características do veículo")

    col_esq, col_dir = st.columns([1, 1])

    with col_esq:
        idade = st.slider(
            "Idade do veículo (anos)", 0, IDADE_MAX, 5,
            help=f"O modelo é válido para veículos de até {IDADE_MAX} anos.",
        )
        km = st.number_input(
            "Quilometragem (km)", min_value=0, max_value=500_000,
            value=80_000, step=5_000,
        )
        cv = st.number_input(
            "Potência (cv)", min_value=40, max_value=600, value=120, step=5,
        )
        puertas = st.selectbox("Número de portas", [2, 3, 4, 5], index=3)

    with col_dir:
        transmision = st.selectbox(
            "Transmissão", opcoes["transmision"],
            index=opcoes["transmision"].index("manual")
            if "manual" in opcoes["transmision"] else 0,
        )
        combustible = st.selectbox(
            "Combustível", opcoes["combustible"],
            index=opcoes["combustible"].index("diesel")
            if "diesel" in opcoes["combustible"] else 0,
        )
        marca = st.selectbox("Marca", opcoes["marca_agrupada"])

    # ----------------------------------------------------------------------
    # A entrada passa pelo MESMO Pipeline do treinamento.
    # Imputação, padronização, codificação e expansão polinomial são os
    # objetos ajustados no treino — não há regra duplicada para a aplicação.
    # ----------------------------------------------------------------------
    entrada = pd.DataFrame([{
        "idade": idade, "km": km, "cv": cv, "puertas": puertas,
        "transmision": transmision, "combustible": combustible,
        "marca_agrupada": marca,
    }])

    previsao_bruta = float(pipe.predict(entrada)[0])

    # ----------------------------------------------------------------------
    # Piso de previsão.
    #
    # O modelo é linear nos demais preditores e pode produzir valores
    # negativos para combinações de idade alta com quilometragem alta —
    # por exemplo, 18 anos e 250.000 km resultam em -1.753 €. Preço negativo
    # não existe, e a combinação não é implausível: é comum no mercado.
    #
    # A causa é estrutural: a regressão linear não impõe restrição de
    # positividade. O tratamento correto seria ajustar em log (rejeitado por
    # viés de retransformação, ver relatório) ou usar um modelo com suporte
    # positivo. Como paliativo declarado, a previsão é limitada inferiormente
    # ao menor preço observado na base, e o usuário é informado de que a
    # estimativa foi truncada.
    # ----------------------------------------------------------------------
    PISO = float(d_modelo["precio"].min())
    previsao = max(previsao_bruta, PISO)
    truncada = previsao_bruta < PISO

    st.divider()

    # Avisos de extrapolação, verificados ANTES de destacar o valor
    avisos = []
    for var, rotulo, valor in [
        ("idade", "Idade", idade), ("km", "Quilometragem", km),
        ("cv", "Potência", cv), ("puertas", "Portas", puertas),
    ]:
        lo, hi = intervalos[var]["min"], intervalos[var]["max"]
        if valor < lo or valor > hi:
            avisos.append(
                f"**{rotulo}** = {valor:,.0f} está fora do intervalo observado "
                f"no treino ({lo:,.0f} a {hi:,.0f})."
            )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if truncada:
            st.error(
                f"### Estimativa: abaixo de {PISO:,.0f} €\n\n"
                f"O modelo calculou **{previsao_bruta:,.0f} €** para esta "
                f"combinação — um valor negativo, impossível para um preço.\n\n"
                f"A regressão linear não impõe restrição de positividade: com "
                f"idade e quilometragem elevadas simultaneamente, os "
                f"coeficientes negativos superam o intercepto. A estimativa foi "
                f"limitada ao menor preço observado na base "
                f"({PISO:,.0f} €), mas **não é confiável** — trate-a apenas "
                f"como indicação de veículo de valor residual."
            )
        else:
            st.success(f"### Preço estimado: {previsao:,.0f} €")
            faixa_lo = max(PISO, previsao - met["MAE"])
            faixa_hi = previsao + met["MAE"]
            st.markdown(
                f"Considerando o erro médio do modelo, a faixa plausível é de "
                f"**{faixa_lo:,.0f} €** a **{faixa_hi:,.0f} €**."
            )

    with col_b:
        # Confiabilidade depende da faixa de preço prevista
        erro_faixa = None
        for f in pacote["erro_por_faixa"]:
            if f["min"] <= previsao <= f["max"]:
                erro_faixa = f["erro_pct"]
                break
        if erro_faixa is None:
            erro_faixa = pacote["erro_por_faixa"][-1]["erro_pct"]

        if erro_faixa > 30:
            st.error(
                f"**Confiabilidade baixa nesta faixa de preço.**\n\n"
                f"Para veículos nesta faixa, o erro mediano do modelo é de "
                f"cerca de **{erro_faixa:.0f}%** do preço. Veículos de menor "
                f"valor dependem fortemente de estado de conservação e "
                f"histórico, informações ausentes na base."
            )
        elif erro_faixa > 20:
            st.warning(
                f"**Confiabilidade moderada.** Erro mediano de cerca de "
                f"**{erro_faixa:.0f}%** do preço nesta faixa."
            )
        else:
            st.info(
                f"**Confiabilidade típica.** Erro mediano de cerca de "
                f"**{erro_faixa:.0f}%** do preço nesta faixa."
            )

    if avisos:
        st.warning("**Atenção — extrapolação:**\n\n" + "\n\n".join(f"- {a}" for a in avisos))

    if cv > 300:
        st.warning(
            "**Potência acima de 300 cv.** O diagnóstico identificou que os "
            "maiores erros do modelo se concentram em veículos de alta "
            "potência: o efeito da potência é tratado como linear, o que não "
            "descreve bem o segmento esportivo e premium."
        )

    # Posição da previsão na distribuição real
    st.divider()
    st.markdown("**Onde esta estimativa se situa na base**")
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.hist(d_modelo["precio"], bins=45, color=AZUL, alpha=0.75, edgecolor="white",
            linewidth=0.4)
    if previsao > 0:
        ax.axvline(previsao, color=LARANJA, lw=2.4,
                   label=f"estimativa: {previsao:,.0f} €")
        pct = 100 * (d_modelo["precio"] < previsao).mean()
        ax.legend(fontsize=8)
        st.caption(f"A estimativa é superior a {pct:.0f}% dos anúncios da base.")
    ax.set_xlabel("Preço anunciado (€)")
    ax.set_ylabel("Nº de anúncios")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==========================================================================
# ABA 2 — Base de dados
# ==========================================================================
with aba_dados:
    st.subheader("Amostra da base tratada")
    st.caption(
        f"{len(d)} anúncios após tratamento (base original: 500). "
        f"O modelo utiliza os {len(d_modelo)} com até {IDADE_MAX} anos."
    )

    colunas_exibir = ["marca_agrupada", "idade", "km", "cv", "puertas",
                      "transmision", "combustible", "precio"]
    st.dataframe(d[colunas_exibir].head(20), use_container_width=True, hide_index=True)

    st.subheader("Estatísticas descritivas")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Variáveis quantitativas**")
        desc = d_modelo[["precio", "idade", "km", "cv", "puertas"]].describe().T
        desc = desc[["count", "mean", "std", "min", "50%", "max"]]
        desc.columns = ["n", "média", "desvio", "mín", "mediana", "máx"]
        st.dataframe(desc.round(1), use_container_width=True)

    with col2:
        st.markdown("**Variáveis categóricas**")
        for c in CAT:
            vc = d_modelo[c].value_counts().head(6)
            st.markdown(f"*{c}*")
            st.dataframe(
                pd.DataFrame({"categoria": vc.index, "anúncios": vc.values}),
                use_container_width=True, hide_index=True, height=min(240, 38 * len(vc) + 38),
            )

    st.info(
        f"**Fonte.** *Coches de segunda mano Milanuncios* — Zenodo, "
        f"DOI [10.5281/zenodo.4674757]({FONTE_URL}), licença CC BY-NC-SA 4.0. "
        f"Extração de 09/04/2021 do portal milanuncios.com."
    )


# ==========================================================================
# ABA 3 — Exploração
# ==========================================================================
with aba_explorar:
    st.subheader("Gráfico 1 — A depreciação não é linear")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(d_modelo["idade"], d_modelo["precio"], s=16, alpha=0.4,
               color=AZUL, edgecolors="none", label="anúncio")
    faixas = pd.cut(d_modelo["idade"], [-0.5, 0.5, 2, 4, 6, 9, 13, 20])
    med = d_modelo.groupby(faixas, observed=True).agg(
        idade=("idade", "median"), precio=("precio", "median"))
    ax.plot(med["idade"], med["precio"], "o-", color="black", ms=5, lw=1.5,
            label="mediana por faixa")
    ax.set_xlabel("Idade do veículo (anos)")
    ax.set_ylabel("Preço anunciado (€)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        "A queda é acentuada nos primeiros anos e se achata a partir dos dez. "
        "A correlação de Pearson entre idade e preço é de −0,47, mas a de "
        "Spearman chega a −0,71: a associação é forte e consistentemente "
        "decrescente, e Pearson a subestima porque a relação não tem forma de "
        "reta. **É essa diferença que justifica o termo quadrático.**"
    )

    st.divider()
    st.subheader("Gráfico 2 — Câmbio automático: efeito real e efeito aparente")

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    grupos = ["manual", "automatico"]
    for i, (var, rotulo) in enumerate(
            [("precio", "Preço (€)"), ("idade", "Idade (anos)"), ("cv", "Potência (cv)")]):
        dados = [d_modelo[d_modelo["transmision"] == g][var].dropna() for g in grupos]
        bp = ax[i].boxplot(dados, tick_labels=grupos, patch_artist=True, widths=0.5)
        for p, c in zip(bp["boxes"], [AZUL, LARANJA]):
            p.set_facecolor(c); p.set_alpha(0.65)
        for m in bp["medians"]:
            m.set_color("black"); m.set_linewidth(1.5)
        ax[i].set_ylabel(rotulo)
        ax[i].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        "A diferença bruta de preço entre automáticos e manuais é de cerca de "
        "**11.000 €**. Mas os painéis ao lado mostram que os automáticos "
        "também são mais novos e mais potentes. Quando o modelo mantém idade, "
        "quilometragem, potência e marca constantes, o prêmio do câmbio "
        "automático cai para cerca de **1.500 €**. "
        "**Aproximadamente 86% da diferença aparente não vinha do câmbio.**"
    )


# ==========================================================================
# ABA 4 — Desempenho detalhado
# ==========================================================================
with aba_desempenho:
    y_test = np.array(pacote["y_test"])
    pred_test = np.array(pacote["pred_test"])
    resid = y_test - pred_test

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Valores reais vs previstos")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.scatter(y_test, pred_test, s=20, alpha=0.5, color=AZUL, edgecolors="none")
        lim = [0, max(y_test.max(), pred_test.max()) * 1.05]
        ax.plot(lim, lim, color=LARANJA, lw=1.8, ls="--", label="previsão perfeita")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Preço real (€)"); ax.set_ylabel("Preço previsto (€)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Os pontos acompanham a diagonal até cerca de 30.000 €. Acima "
            "disso concentram-se abaixo dela: o modelo subestima os veículos "
            "mais caros."
        )

    with col2:
        st.subheader("Resíduos vs valores ajustados")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.scatter(pred_test, resid, s=20, alpha=0.5, color=AZUL, edgecolors="none")
        ax.axhline(0, color=LARANJA, lw=1.8, ls="--")
        ax.set_xlabel("Valor previsto (€)"); ax.set_ylabel("Resíduo (€)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "A dispersão aumenta com o valor previsto — heteroscedasticidade "
            "(Spearman entre |resíduo| e ajustado: ρ = 0,22; p = 0,008). O "
            "pressuposto de variância constante não se sustenta."
        )

    st.divider()
    st.subheader("Erro relativo por faixa de preço")

    faixas_erro = pacote["erro_por_faixa"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rotulos = [f"{f['min']/1000:.0f}–{f['max']/1000:.0f}k" for f in faixas_erro]
    valores = [f["erro_pct"] for f in faixas_erro]
    cores = [LARANJA if v > 30 else AZUL for v in valores]
    ax.bar(range(len(valores)), valores, color=cores, alpha=0.85)
    ax.set_xticks(range(len(valores))); ax.set_xticklabels(rotulos)
    ax.set_xlabel("Faixa de preço real (€)")
    ax.set_ylabel("Erro mediano (% do preço)")
    for i, v in enumerate(valores):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8.5)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.warning(
        "**Limitação mais relevante da aplicação.** Em euros, o modelo erra "
        "mais nos carros caros. Em percentual, erra muito mais nos baratos: "
        "cerca de metade do preço na faixa mais baixa. Veículos de menor valor "
        "dependem de estado de conservação, histórico de manutenção e urgência "
        "do vendedor — nenhum deles presente na base."
    )

    with st.expander("Quando o modelo não deve ser utilizado"):
        st.markdown(f"""
- **Veículos com mais de {IDADE_MAX} anos.** Fora do domínio de treinamento.
  Clássicos são precificados por raridade, não por depreciação.
- **Veículos abaixo de 6.000 €.** Erro relativo mediano próximo de 50%.
- **Alta potência (acima de 300 cv).** Concentram os maiores erros absolutos.
- **Marcas premium antigas.** O modelo soma prêmio de marca e de potência sem
  captar a interação com idade avançada.
- **Fora do mercado espanhol.** Os dados são de 09/04/2021, em euros. Não há
  base para transpor as conclusões a outro país ou período.
- **Para decisão de compra ou venda isolada.** A previsão estima o preço de
  **anúncio**, não o de transação, e não substitui avaliação presencial.
        """)

st.divider()

with st.expander("Integrantes do grupo"):
    st.markdown("""
| Nome | RM |
|---|---|
| Leonardo Eiji Kina | 562784 |
| Nicholas Braga de Souza | 561733 |
| Tomé Rossi Giani | 562422 |
| Vitor Ramos de Farias | 561958 |
    """)

st.caption(
    "Checkpoint 4 — Data Science & Statistical Computing — FIAP 2026. "
    f"Dados: Milanuncios via Zenodo (DOI 10.5281/zenodo.4674757), CC BY-NC-SA 4.0. "
    f"Modelo treinado com scikit-learn {pacote['versao_sklearn']}."
)
