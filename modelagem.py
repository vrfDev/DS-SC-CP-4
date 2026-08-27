"""
Modelagem — Checkpoint 4

Os quatro modelos exigidos pelo enunciado:
  1. Referência (média da resposta)
  2. Regressão linear simples (um preditor)
  3. Regressão linear múltipla
  4. Regressão polinomial

CONTROLE DE VAZAMENTO
---------------------
Toda transformação aprendida (imputação, padronização, one-hot, expansão
polinomial) está dentro de um Pipeline. O Pipeline é ajustado com .fit()
somente em X_train. O conjunto de teste passa apenas por .transform(),
com os parâmetros estimados no treino.

Consequência prática: a mediana usada para imputar `cv` é a mediana do
TREINO, não da base completa. Se fosse calculada antes do split, informação
do teste contaminaria o pré-processamento e as métricas ficariam otimistas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.30

NUM = ["idade", "km", "cv", "puertas"]
CAT = ["transmision", "combustible", "marca_agrupada"]


def metricas(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def bloco_numerico():
    """Imputação por mediana + padronização. Ambas ajustadas só no treino."""
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])


def bloco_categorico():
    return Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore",
                                    drop="first", sparse_output=False)),
    ])


def construir(tipo):
    """Devolve o Pipeline completo de cada modelo."""
    if tipo == "referencia":
        # Prevê sempre a média do TREINO. É o piso: qualquer modelo útil
        # precisa superá-lo. Por construção, seu R² no teste fica próximo de 0.
        return Pipeline([("modelo", DummyRegressor(strategy="mean"))]), ["idade"]

    if tipo == "simples":
        # Preditor único: cv.
        # Justificativa: é o preditor com maior correlação linear com o preço
        # (r = 0,617), acima de idade (-0,471) e km (-0,475). A escolha é
        # deliberada — idade tem associação monótona mais forte (Spearman
        # -0,712), mas sua relação com o preço não é linear, o que a torna
        # inadequada para um modelo linear de um só preditor.
        pre = ColumnTransformer([("num", bloco_numerico(), ["cv"])])
        return Pipeline([("pre", pre), ("modelo", LinearRegression())]), ["cv"]

    if tipo == "multipla":
        pre = ColumnTransformer([
            ("num", bloco_numerico(), NUM),
            ("cat", bloco_categorico(), CAT),
        ])
        return Pipeline([("pre", pre), ("modelo", LinearRegression())]), NUM + CAT

    if tipo == "polinomial":
        # Grau 2 aplicado APENAS a `idade`, não a todos os preditores.
        #
        # Justificativa do grau: a Figura 2 mostra queda acentuada seguida de
        # achatamento — uma única mudança de curvatura, que o grau 2 descreve.
        # O grau 3 permitiria uma segunda inflexão sem evidência que a sustente
        # e aumentaria a instabilidade nas extremidades, onde há menos dados.
        #
        # Justificativa do escopo: expandir todos os preditores geraria termos
        # cruzados e quadráticos para km, cv e puertas, elevando muito a
        # dimensão para n = 484. A evidência de não linearidade é específica de
        # `idade`.
        poli = Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("expandir", PolynomialFeatures(degree=2, include_bias=False)),
            ("escalar", StandardScaler()),
        ])
        outras = [c for c in NUM if c != "idade"]
        pre = ColumnTransformer([
            ("idade_poli", poli, ["idade"]),
            ("num", bloco_numerico(), outras),
            ("cat", bloco_categorico(), CAT),
        ])
        return Pipeline([("pre", pre), ("modelo", LinearRegression())]), NUM + CAT

    raise ValueError(tipo)


def avaliar(df, rotulo, usar_log=False):
    """Ajusta os quatro modelos e devolve as métricas no conjunto de teste."""
    X = df[NUM + CAT]
    y = df["precio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if usar_log:
        y_train_aj, y_test_aj = np.log(y_train), np.log(y_test)
    else:
        y_train_aj, y_test_aj = y_train, y_test

    linhas = []
    for tipo in ["referencia", "simples", "multipla", "polinomial"]:
        pipe, cols = construir(tipo)
        pipe.fit(X_train[cols], y_train_aj)
        pred = pipe.predict(X_test[cols])

        # Métricas sempre em euros: se o modelo foi ajustado em log,
        # a previsão é retransformada antes de comparar. Comparar métricas
        # em escalas diferentes seria inválido.
        if usar_log:
            pred = np.exp(pred)

        m = metricas(y_test, pred)
        m["Modelo"] = tipo
        m["Cenário"] = rotulo
        linhas.append(m)

    return pd.DataFrame(linhas)[["Cenário", "Modelo", "MAE", "RMSE", "R2"]], \
           (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    d = pd.read_csv("dados/coches_tratado.csv")

    print("=" * 78)
    print("CENÁRIO A — base completa (484 anúncios, clássicos incluídos)")
    print("=" * 78)
    a, _ = avaliar(d, "A: completa")
    print(a.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n" + "=" * 78)
    print("CENÁRIO B — domínio restrito a veículos com até 20 anos")
    print("=" * 78)
    d_rest = d[d.possivel_classico == 0].copy()
    print(f"(exclui {len(d) - len(d_rest)} veículos sinalizados como possíveis clássicos)")
    b, _ = avaliar(d_rest, "B: <=20 anos")
    print(b.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n" + "=" * 78)
    print("CENÁRIO C — domínio restrito + resposta em log")
    print("=" * 78)
    c, _ = avaliar(d_rest, "C: log", usar_log=True)
    print(c.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    todos = pd.concat([a, b, c], ignore_index=True)
    todos.to_csv("resultados_modelos.csv", index=False)
    print("\nSalvo em resultados_modelos.csv")
