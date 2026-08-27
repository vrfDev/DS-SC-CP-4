"""
Diagnóstico do modelo final — Checkpoint 4 (seção 3.8)

Avalia os pressupostos da regressão linear no modelo selecionado:
linearidade, homoscedasticidade, normalidade aproximada dos resíduos,
independência dos erros, colinearidade e observações influentes.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

from modelagem import construir, NUM, CAT, RANDOM_STATE, TEST_SIZE

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
AZUL, LARANJA, CINZA = "#2c6fbb", "#e07b39", "#7a7a7a"

d = pd.read_csv("dados/coches_tratado.csv")
base = d[d.possivel_classico == 0].copy()

X, y = base[NUM + CAT], base["precio"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

MODELO = "polinomial"
pipe, cols = construir(MODELO)
pipe.fit(X_train[cols], y_train)

pred = pipe.predict(X_test[cols])
resid = y_test.values - pred

# ======================================================================
# VIF — colinearidade entre os preditores quantitativos
# ======================================================================
# VIF_j = 1 / (1 - R²_j), onde R²_j é obtido regredindo o preditor j
# contra os demais. Mede quanto a variância do coeficiente é inflada
# pela redundância com os outros preditores.
from sklearn.linear_model import LinearRegression

quant = X_train[NUM].copy()
quant = quant.fillna(quant.median())

print("=" * 66)
print("FATOR DE INFLAÇÃO DA VARIÂNCIA (VIF)")
print("=" * 66)
vifs = {}
for v in NUM:
    outros = [c for c in NUM if c != v]
    r2 = LinearRegression().fit(quant[outros], quant[v]).score(quant[outros], quant[v])
    vif = 1 / (1 - r2) if r2 < 1 else np.inf
    vifs[v] = vif
    marca = "  <- ATENÇÃO" if vif > 5 else ""
    print(f"  {v:8s}: VIF = {vif:5.2f}{marca}")
print("\n  Referência usual: VIF > 5 indica colinearidade relevante;")
print("  VIF > 10, colinearidade severa.")

# ======================================================================
# FIGURA 6 — Painel de diagnóstico (4 gráficos exigidos)
# ======================================================================
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))

# (a) Real vs previsto
a = ax[0, 0]
a.scatter(y_test, pred, s=20, alpha=0.5, color=AZUL, edgecolors="none")
lim = [0, max(y_test.max(), pred.max()) * 1.05]
a.plot(lim, lim, color=LARANJA, lw=1.8, ls="--", label="previsão perfeita")
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("Preço real (€)"); a.set_ylabel("Preço previsto (€)")
a.set_title("(a) Valores reais vs previstos")
a.legend(fontsize=8)

# (b) Resíduos vs ajustados
b = ax[0, 1]
b.scatter(pred, resid, s=20, alpha=0.5, color=AZUL, edgecolors="none")
b.axhline(0, color=LARANJA, lw=1.8, ls="--")
# linha de tendência da dispersão absoluta, para enxergar heteroscedasticidade
ordem = np.argsort(pred)
jan = max(5, len(pred) // 10)
suave = pd.Series(np.abs(resid)[ordem]).rolling(jan, center=True, min_periods=3).mean()
b.plot(pred[ordem], suave, color="black", lw=1.5, label="|resíduo| médio móvel")
b.plot(pred[ordem], -suave, color="black", lw=1.5)
b.set_xlabel("Valor previsto (€)"); b.set_ylabel("Resíduo (€)")
b.set_title("(b) Resíduos vs valores ajustados")
b.legend(fontsize=8)

# (c) Distribuição dos resíduos
c = ax[1, 0]
c.hist(resid, bins=30, color=AZUL, edgecolor="white", linewidth=0.5, density=True)
xx = np.linspace(resid.min(), resid.max(), 200)
c.plot(xx, stats.norm.pdf(xx, resid.mean(), resid.std()),
       color=LARANJA, lw=2, label="normal ajustada")
c.set_xlabel("Resíduo (€)"); c.set_ylabel("Densidade")
c.set_title(f"(c) Distribuição dos resíduos\nassimetria = {stats.skew(resid):.2f}")
c.legend(fontsize=8)

# (d) QQ-plot
dax = ax[1, 1]
stats.probplot(resid, dist="norm", plot=dax)
dax.get_lines()[0].set_markerfacecolor(AZUL)
dax.get_lines()[0].set_markeredgecolor("none")
dax.get_lines()[0].set_markersize(4)
dax.get_lines()[1].set_color(LARANJA)
dax.get_lines()[1].set_linewidth(2)
dax.set_xlabel("Quantis teóricos (normal)")
dax.set_ylabel("Quantis observados (€)")
dax.set_title("(d) QQ-plot dos resíduos")

fig.suptitle(f"Figura 6 — Diagnóstico do modelo {MODELO} (conjunto de teste, n = {len(y_test)})",
             fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig("figuras/fig6_diagnostico.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# Testes formais e erros extremos
# ======================================================================
print("\n" + "=" * 66)
print("PRESSUPOSTOS")
print("=" * 66)

print(f"\nResíduo médio: {resid.mean():,.0f} €  (esperado próximo de 0)")
print(f"Desvio-padrão: {resid.std():,.0f} €")
print(f"Assimetria   : {stats.skew(resid):.2f}")
print(f"Curtose      : {stats.kurtosis(resid):.2f}")

sh = stats.shapiro(resid)
print(f"\nShapiro-Wilk (normalidade): W = {sh.statistic:.4f}, p = {sh.pvalue:.2e}")
print("  p < 0,05 rejeita normalidade dos resíduos.")

# Heteroscedasticidade: correlação entre |resíduo| e valor ajustado
rho, pval = stats.spearmanr(pred, np.abs(resid))
print(f"\nSpearman(|resíduo|, ajustado): rho = {rho:.3f}, p = {pval:.2e}")
print("  rho positivo e significativo indica variância crescente (heteroscedasticidade).")

print("\n" + "=" * 66)
print("MAIORES ERROS")
print("=" * 66)
res = pd.DataFrame({
    "real": y_test.values, "previsto": pred.round(0), "erro": resid.round(0),
    "idade": X_test.idade.values, "km": X_test.km.values,
    "cv": X_test.cv.values, "marca": X_test.marca_agrupada.values,
})
piores = res.reindex(res.erro.abs().sort_values(ascending=False).index).head(6)
print(piores.to_string(index=False))

print(f"\nErro absoluto mediano: {np.median(np.abs(resid)):,.0f} €")
print(f"Erro absoluto médio  : {np.mean(np.abs(resid)):,.0f} €")
print(f"90% dos erros abaixo de: {np.percentile(np.abs(resid), 90):,.0f} €")

# ======================================================================
# FIGURA 7 — Erro relativo por faixa de preço
# ======================================================================
fig, ax = plt.subplots(figsize=(6.5, 4))
faixas = pd.qcut(y_test, 5)
erro_rel = pd.DataFrame({"faixa": faixas.values,
                         "erro_pct": 100 * np.abs(resid) / y_test.values})
g = erro_rel.groupby("faixa", observed=True)["erro_pct"].median()
rot = [f"{int(i.left/1000)}–{int(i.right/1000)}k" for i in g.index]
ax.bar(range(len(g)), g.values, color=AZUL, alpha=0.85)
ax.set_xticks(range(len(g))); ax.set_xticklabels(rot)
ax.set_xlabel("Faixa de preço real (€)")
ax.set_ylabel("Erro absoluto mediano (% do preço)")
ax.set_title("Figura 7 — O erro relativo é maior nos carros baratos",
             fontweight="bold", fontsize=10)
for i, v in enumerate(g.values):
    ax.text(i, v + 0.4, f"{v:.0f}%", ha="center", fontsize=8.5)
fig.tight_layout()
fig.savefig("figuras/fig7_erro_relativo.png", bbox_inches="tight")
plt.close(fig)

print("\nFiguras 6 e 7 geradas.")
