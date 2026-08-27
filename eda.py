"""
Análise exploratória — Checkpoint 4
Gera as figuras da seção 3.4. Cada figura responde a uma pergunta específica.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
AZUL, LARANJA, CINZA = "#2c6fbb", "#e07b39", "#7a7a7a"

d = pd.read_csv("dados/coches_tratado.csv")

# ======================================================================
# FIGURA 1 — Distribuição da variável resposta
# Pergunta: o preço é simétrico? Precisa de transformação?
# ======================================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

ax[0].hist(d.precio, bins=40, color=AZUL, edgecolor="white", linewidth=0.5)
ax[0].axvline(d.precio.median(), color=LARANJA, lw=2,
              label=f"mediana = {d.precio.median():,.0f} €")
ax[0].axvline(d.precio.mean(), color=CINZA, lw=2, ls="--",
              label=f"média = {d.precio.mean():,.0f} €")
ax[0].set_xlabel("Preço anunciado (€)")
ax[0].set_ylabel("Nº de anúncios")
ax[0].set_title(f"Preço em escala original\nassimetria = {d.precio.skew():.2f}")
ax[0].legend(fontsize=7.5)

ax[1].hist(np.log(d.precio), bins=40, color=AZUL, edgecolor="white", linewidth=0.5)
ax[1].set_xlabel("log(Preço) — log de euros")
ax[1].set_ylabel("Nº de anúncios")
ax[1].set_title(f"Preço em escala logarítmica\nassimetria = {np.log(d.precio).skew():.2f}")

fig.suptitle("Figura 1 — Distribuição do preço anunciado (n = %d)" % len(d),
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("figuras/fig1_distribuicao_preco.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 2 — Preço vs idade: o teste da hipótese H1
# Pergunta: a depreciação é linear?
# ======================================================================
fig, ax = plt.subplots(figsize=(7, 4.4))

ax.scatter(d.idade, d.precio, s=18, alpha=0.45, color=AZUL,
           edgecolors="none", label="anúncio")

x = np.linspace(d.idade.min(), d.idade.max(), 200)
b1 = np.polyfit(d.idade, d.precio, 1)
b2 = np.polyfit(d.idade, d.precio, 2)
ax.plot(x, np.polyval(b1, x), color=CINZA, lw=2, ls="--", label="ajuste linear")
ax.plot(x, np.polyval(b2, x), color=LARANJA, lw=2.2, label="ajuste quadrático")

# medianas por faixa mostram a forma real, sem depender de modelo
faixas = pd.cut(d.idade, [-0.5, 0.5, 2, 4, 6, 9, 13, 20, 30])
med = d.groupby(faixas, observed=True).agg(
    idade=("idade", "median"), precio=("precio", "median"))
ax.plot(med.idade, med.precio, "o-", color="black", ms=5, lw=1.4,
        label="mediana por faixa")

classicos = d[d.possivel_classico == 1]
ax.scatter(classicos.idade, classicos.precio, s=70, facecolors="none",
           edgecolors="red", linewidths=1.4, label="possível clássico (>20 anos)")

ax.set_xlabel("Idade do veículo (anos)")
ax.set_ylabel("Preço anunciado (€)")
ax.set_title("Figura 2 — Preço por idade: a depreciação não é linear",
             fontweight="bold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figuras/fig2_preco_idade.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 3 — Correlações
# Pergunta: quais preditores importam, e há colinearidade?
# ======================================================================
quant = ["precio", "idade", "km", "cv", "puertas"]
corr = d[quant].corr()

fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(quant))); ax.set_xticklabels(quant, rotation=45, ha="right")
ax.set_yticks(range(len(quant))); ax.set_yticklabels(quant)
for i in range(len(quant)):
    for j in range(len(quant)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(v) > 0.55 else "black")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="correlação de Pearson")
ax.set_title("Figura 3 — Correlações entre variáveis quantitativas",
             fontweight="bold", fontsize=9.5)
fig.tight_layout()
fig.savefig("figuras/fig3_correlacoes.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 4 — Categórica: transmissão e o risco de confundimento
# Pergunta: automático é caro por ser automático, ou por ser mais novo?
# ======================================================================
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))

grupos = ["manual", "automatico"]
dados = [d[d.transmision == g].precio for g in grupos]
bp = ax[0].boxplot(dados, tick_labels=grupos, patch_artist=True, widths=0.5)
for p, c in zip(bp["boxes"], [AZUL, LARANJA]):
    p.set_facecolor(c); p.set_alpha(0.65)
for m in bp["medians"]:
    m.set_color("black"); m.set_linewidth(1.6)
ax[0].set_ylabel("Preço anunciado (€)")
ax[0].set_title("Preço por transmissão")

for i, (var, rot) in enumerate([("idade", "Idade (anos)"), ("cv", "Potência (cv)")], start=1):
    dd = [d[d.transmision == g][var].dropna() for g in grupos]
    bp = ax[i].boxplot(dd, tick_labels=grupos, patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], [AZUL, LARANJA]):
        p.set_facecolor(c); p.set_alpha(0.65)
    for m in bp["medians"]:
        m.set_color("black"); m.set_linewidth(1.6)
    ax[i].set_ylabel(rot)
    ax[i].set_title(f"{var} por transmissão")

fig.suptitle("Figura 4 — Automáticos custam mais, mas também são mais novos e potentes",
             fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig("figuras/fig4_transmissao.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURA 5 — Colinearidade km x idade (hipótese H3)
# ======================================================================
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.scatter(d.idade, d.km / 1000, s=18, alpha=0.45, color=AZUL, edgecolors="none")
b = np.polyfit(d.idade, d.km / 1000, 1)
xx = np.linspace(d.idade.min(), d.idade.max(), 50)
ax.plot(xx, np.polyval(b, xx), color=LARANJA, lw=2)
r = d.idade.corr(d.km)
ax.set_xlabel("Idade do veículo (anos)")
ax.set_ylabel("Quilometragem (mil km)")
ax.set_title(f"Figura 5 — Idade e quilometragem são redundantes (r = {r:.2f})",
             fontweight="bold", fontsize=9.5)
fig.tight_layout()
fig.savefig("figuras/fig5_colinearidade.png", bbox_inches="tight")
plt.close(fig)

print("5 figuras geradas em figuras/")
