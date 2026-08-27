"""
Data wrangling — Checkpoint 4
Base: Coches de segunda mano Milanuncios (Zenodo, DOI 10.5281/zenodo.4674757)

PRINCÍPIO METODOLÓGICO
----------------------
Este script executa APENAS limpeza determinística: operações cujo resultado
não depende de nenhuma estatística estimada a partir dos dados (parsing de
texto para número, remoção de duplicatas, marcação de valores impossíveis,
padronização de categorias).

Transformações APRENDIDAS (imputação por mediana, padronização, one-hot
encoding, expansão polinomial) NÃO estão aqui. Elas vão para o Pipeline,
ajustado somente no conjunto de treino, para evitar vazamento de dados.

Essa separação é a razão de o arquivo existir: aplicar imputação aqui
contaminaria o conjunto de teste com informação do conjunto completo.
"""

import pandas as pd
import numpy as np

ANO_REFERENCIA = 2021  # ver DECISÃO 2

# Registro de todas as decisões, para a tabela do relatório
log = []


def registrar(problema, decisao, afetados):
    log.append({"Problema": problema, "Decisão": decisao, "Obs. afetadas": afetados})


def para_numero(serie):
    """Converte texto espanhol para número.

    O ponto é separador de MILHAR ('3.900' = 3900, '241.665' = 241665).
    Converter sem tratar produziria preços de 3,90 € e carros com 241 km.
    """
    return pd.to_numeric(
        serie.astype(str).str.replace(".", "", regex=False).str.strip(),
        errors="coerce",
    )


def limpar(caminho_csv):
    # Lê tudo como texto: a inferência automática do pandas interpretaria
    # '3.900' como float 3.9
    df = pd.read_csv(caminho_csv, dtype=str)
    n_inicial = len(df)

    # ------------------------------------------------------------------
    # DECISÃO 1 — Duplicatas de conteúdo
    # ------------------------------------------------------------------
    # Não há linhas totalmente idênticas, mas há veículos repetidos com
    # referência de anúncio diferente: mesma marca, modelo, ano, km, preço e
    # potência. Exemplo: OPEL CROSSLAND X 2020 aparece 6 vezes.
    #
    # Interpretação: concessionária anunciando várias unidades do mesmo
    # modelo em estoque. Não é erro de coleta.
    #
    # Por que remover: cada unidade repetida não traz informação nova sobre a
    # relação entre características e preço. Mantê-las dá peso maior a esses
    # veículos no ajuste e viola a independência das observações — pressuposto
    # avaliado no diagnóstico do modelo.
    chave = ["marca", "modelo", "ano_vehic", "km", "precio", "cv"]
    n_dup = df.duplicated(subset=chave).sum()
    df = df.drop_duplicates(subset=chave, keep="first").copy()
    registrar(
        "Veículos repetidos (mesmo modelo/ano/km/preço/cv, anúncios distintos)",
        "Mantida a primeira ocorrência de cada veículo",
        n_dup,
    )

    # ------------------------------------------------------------------
    # Conversão de tipos
    # ------------------------------------------------------------------
    df["precio"] = para_numero(df["precio"])
    df["km"] = para_numero(df["km"])
    df["ano_vehic"] = pd.to_numeric(df["ano_vehic"], errors="coerce")
    df["cv"] = pd.to_numeric(df["cv"], errors="coerce")
    df["puertas"] = pd.to_numeric(df["puertas"], errors="coerce")
    registrar(
        "precio e km armazenados como texto com ponto de milhar",
        "Removido o separador e convertido para numérico",
        n_inicial,
    )

    # ------------------------------------------------------------------
    # DECISÃO 2 — Ano de referência
    # ------------------------------------------------------------------
    # A documentação da fonte é contraditória: a descrição no Zenodo informa
    # extração em 09/04/2020, mas o arquivo se chama '..._09_04_2021.csv' e a
    # publicação é de 09/04/2021.
    #
    # Evidência decisiva nos dados: existem 16 veículos com ano_vehic = 2021,
    # impossíveis em uma extração de abril de 2020.
    #
    # Decisão: ANO_REFERENCIA = 2021. Sob 2020, esses 16 veículos teriam idade
    # negativa.
    df["idade"] = ANO_REFERENCIA - df["ano_vehic"]
    registrar(
        "Data de referência ambígua na documentação da fonte",
        f"Adotado {ANO_REFERENCIA}, evidenciado por veículos ano 2021 na base",
        16,
    )

    # ------------------------------------------------------------------
    # DECISÃO 3 — Padronização de categorias
    # ------------------------------------------------------------------
    df["transmision"] = df["transmision"].str.strip().str.lower()
    df["combustible"] = df["combustible"].str.strip().str.lower()
    df["marca"] = df["marca"].str.strip().str.upper()
    # 'automat' é abreviação da fonte para automático; renomeado por clareza
    df["transmision"] = df["transmision"].replace({"automat": "automatico"})
    registrar(
        "Categorias com espaços e caixa inconsistentes; 'automat' abreviado",
        "Padronizadas caixa e espaços; 'automat' renomeado para 'automatico'",
        len(df),
    )

    # NOTA PARA O RELATÓRIO: a documentação da fonte declara que combustible
    # assume apenas gasolina/diesel, mas a base contém 57 registros 'híbrido'.
    # A documentação está incompleta — mantidos os três níveis.

    # ------------------------------------------------------------------
    # DECISÃO 4 — Valores ausentes de km NÃO são aleatórios
    # ------------------------------------------------------------------
    # Os 17 ausentes de km concentram-se integralmente em veículos 2020 e 2021
    # (ano médio 2020,6 contra 2014,1 nos registros com km preenchido).
    #
    # Interpretação: são veículos zero-quilômetro ou de demonstração, em que a
    # loja não declara quilometragem.
    #
    # Consequência prática: imputar pela mediana global (85.000 km) atribuiria
    # a um carro novo o uso de um veículo de sete anos. A ausência aqui é
    # informativa, não é dado perdido.
    #
    # Decisão: imputar 0 para veículos com idade <= 1; os demais casos seguem
    # para imputação por mediana no Pipeline. Como a regra depende apenas da
    # idade do próprio registro, e não de estatística estimada da base, ela é
    # determinística e pode ser aplicada aqui sem vazamento.
    novo_sem_km = df["km"].isna() & (df["idade"] <= 1)
    n_novo = int(novo_sem_km.sum())
    df.loc[novo_sem_km, "km"] = 0
    registrar(
        "km ausente concentrado em veículos 2020/2021 (zero-km, ausência informativa)",
        "Imputado km = 0 para veículos com idade <= 1 ano",
        n_novo,
    )

    # Marcador de veículo essencialmente novo — pode ser útil na modelagem e
    # documenta a decisão acima de forma auditável
    df["km_era_ausente"] = novo_sem_km.astype(int)

    # ------------------------------------------------------------------
    # DECISÃO 5 — Valores impossíveis
    # ------------------------------------------------------------------
    # Idade negativa não pode existir. Sob ANO_REFERENCIA = 2021 não deve
    # haver nenhum caso; a verificação permanece como proteção caso o ano de
    # referência seja alterado.
    impossivel = df["idade"] < 0
    n_imp = int(impossivel.sum())
    df = df.loc[~impossivel].copy()
    registrar("Idade negativa (ano do veículo posterior à referência)",
              "Registros removidos", n_imp)

    # Preço ou idade ausentes: são a variável resposta e o preditor principal.
    # Imputar a resposta inventaria o alvo do modelo.
    sem_alvo = df["precio"].isna() | df["idade"].isna()
    n_alvo = int(sem_alvo.sum())
    df = df.loc[~sem_alvo].copy()
    registrar("precio ou idade ausentes",
              "Removidos: imputar a variável resposta não é admissível", n_alvo)

    # ------------------------------------------------------------------
    # DECISÃO 6 — Marcas raras
    # ------------------------------------------------------------------
    # São 34 marcas, das quais 16 têm menos de 5 anúncios. Com ~480 registros,
    # criar uma variável indicadora para cada marca gastaria graus de liberdade
    # em categorias com pouquíssimo suporte, produzindo coeficientes instáveis.
    #
    # O corte em 10 é uma escolha do grupo, não uma regra estatística.
    LIMITE = 10
    freq = df["marca"].value_counts()
    raras = freq[freq < LIMITE].index
    n_raras = int(df["marca"].isin(raras).sum())
    df["marca_agrupada"] = df["marca"].where(~df["marca"].isin(raras), "OUTRAS")
    registrar(
        f"{len(raras)} marcas com menos de {LIMITE} anúncios",
        f"Agrupadas na categoria 'OUTRAS' ({df['marca_agrupada'].nunique()} categorias finais)",
        n_raras,
    )

    # ------------------------------------------------------------------
    # DECISÃO 7 — Discrepantes: SINALIZAR, não remover
    # ------------------------------------------------------------------
    # O enunciado veda remoção automática de discrepantes. Veículos com mais
    # de 20 anos apresentam preço mediano ACIMA do de veículos de 14 a 20 anos
    # — comportamento de mercado de colecionador, não erro de registro.
    #
    # São marcados, e não excluídos. A decisão sobre restringir o domínio do
    # modelo é tomada na etapa de modelagem, com justificativa própria.
    df["possivel_classico"] = (df["idade"] > 20).astype(int)
    registrar(
        "Veículos com mais de 20 anos e preço acima da curva de depreciação",
        "SINALIZADOS para análise (não removidos): exceção legítima de mercado",
        int(df["possivel_classico"].sum()),
    )

    # ------------------------------------------------------------------
    # Descarte de colunas fora do escopo (justificado na seção 1.4)
    # ------------------------------------------------------------------
    descartar = ["url", "referencia", "titulo", "vendedor", "descripcion",
                 "particular", "stats_visto", "stats_contactado",
                 "stats_compartido", "stats_favorito", "stats_renovados"]
    df = df.drop(columns=[c for c in descartar if c in df.columns])
    registrar("Colunas sem papel causal ou identificadores",
              "Removidas (ver justificativa na proposição do problema)",
              len(descartar))

    return df, pd.DataFrame(log), n_inicial


if __name__ == "__main__":
    df, registro, n_inicial = limpar("dados/coches.csv")

    print("=" * 72)
    print("REGISTRO DE TRATAMENTOS")
    print("=" * 72)
    print(registro.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"Base original : {n_inicial} linhas")
    print(f"Base tratada  : {len(df)} linhas  "
          f"({len(df) - n_inicial:+d}, {100*len(df)/n_inicial:.1f}% do original)")
    print(f"Colunas       : {df.shape[1]}")
    print("=" * 72)

    print("\nAusentes restantes (a tratar no Pipeline, ajustado só no treino):")
    faltando = df.isna().sum()
    print(faltando[faltando > 0].to_string() if faltando.sum() else "  nenhum")

    print("\nEstatísticas da variável resposta (precio, €):")
    print(df["precio"].describe().round(0).to_string())

    df.to_csv("dados/coches_tratado.csv", index=False)
    print("\nSalvo em dados/coches_tratado.csv")
