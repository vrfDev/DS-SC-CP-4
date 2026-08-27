"""
Treina o modelo final e serializa Pipeline + metadados.

Executar antes de subir a aplicação:
    python treinar_modelo.py

Gera modelo/modelo.pkl contendo o Pipeline ajustado e todos os metadados de
que a aplicação precisa (métricas, intervalos observados, opções de
formulário). A aplicação não recalcula nada disso: ela apenas lê.

MODELO ESCOLHIDO
----------------
Regressão polinomial, grau 2 em `idade`, domínio restrito a veículos com até
20 anos.

Justificativa da restrição de domínio: os seis veículos com mais de 20 anos
identificados na exploração são clássicos, cujo preço responde a raridade e
não a depreciação. Mantidos, puxam o termo quadrático para cima e fazem o
modelo prever cerca de 17.000 euros para um veículo de 27 anos, contra
mediana observada próxima de 5.000. Não foram descartados por conveniência
estatística: foram excluídos do domínio de aplicação, o que é declarado ao
usuário na interface.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30
IDADE_MAXIMA = 20

NUM = ["idade", "km", "cv", "puertas"]
CAT = ["transmision", "combustible", "marca_agrupada"]


def construir_pipeline():
    """Pipeline do modelo final.

    Usa exclusivamente transformadores do scikit-learn, sem funções próprias.
    Isso é deliberado: um Pipeline que dependa de código local exigiria que o
    mesmo módulo estivesse importável no momento de carregar o pickle.
    """
    poli_idade = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("expandir", PolynomialFeatures(degree=2, include_bias=False)),
        ("escalar", StandardScaler()),
    ])
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", drop="first",
                                    sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("idade_poli", poli_idade, ["idade"]),
        ("num", numericas, [c for c in NUM if c != "idade"]),
        ("cat", categoricas, CAT),
    ])
    return Pipeline([("pre", pre), ("modelo", LinearRegression())])


def treinar(caminho_dados="dados/coches_tratado.csv"):
    d = pd.read_csv(caminho_dados)
    d = d[d["idade"] <= IDADE_MAXIMA].copy()

    X = d[NUM + CAT]
    y = d["precio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    pipe = construir_pipeline()
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    metricas = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "erro_mediano": float(np.median(np.abs(y_test - pred))),
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
    }

    # Intervalos observados: usados para avisar sobre extrapolação.
    # Calculados no TREINO, que é o que o modelo efetivamente viu.
    intervalos = {
        c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())}
        for c in NUM
    }

    # Faixa de erro relativo por faixa de preço, para contextualizar a previsão
    erro_rel = pd.DataFrame({
        "real": y_test.values,
        "erro_pct": 100 * np.abs(y_test.values - pred) / y_test.values,
    })
    faixas = pd.qcut(erro_rel["real"], 5)
    erro_por_faixa = [
        {"min": float(i.left), "max": float(i.right), "erro_pct": float(v)}
        for i, v in erro_rel.groupby(faixas, observed=True)["erro_pct"].median().items()
    ]

    pacote = {
        "pipeline": pipe,
        "metricas": metricas,
        "intervalos": intervalos,
        "erro_por_faixa": erro_por_faixa,
        "colunas_num": NUM,
        "colunas_cat": CAT,
        "opcoes": {c: sorted(d[c].dropna().unique().tolist()) for c in CAT},
        "idade_maxima": IDADE_MAXIMA,
        "versao_sklearn": sklearn.__version__,
        # Guardados para os gráficos da aplicação, evitando recomputar
        "y_test": y_test.values.tolist(),
        "pred_test": pred.tolist(),
    }

    Path("modelo").mkdir(exist_ok=True)
    joblib.dump(pacote, "modelo/modelo.pkl", compress=3)

    with open("modelo/metricas.json", "w", encoding="utf-8") as f:
        json.dump({"metricas": metricas, "intervalos": intervalos,
                   "versao_sklearn": sklearn.__version__},
                  f, indent=2, ensure_ascii=False)

    return pacote


if __name__ == "__main__":
    p = treinar()
    m = p["metricas"]
    print("Modelo treinado e salvo em modelo/modelo.pkl")
    print(f"  scikit-learn      : {p['versao_sklearn']}")
    print(f"  treino / teste    : {m['n_treino']} / {m['n_teste']}")
    print(f"  MAE               : {m['MAE']:,.0f} EUR")
    print(f"  RMSE              : {m['RMSE']:,.0f} EUR")
    print(f"  R2                : {m['R2']:.3f}")
    print(f"  erro mediano      : {m['erro_mediano']:,.0f} EUR")
