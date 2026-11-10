
# ETAPA 5 - PASSO 20
# Consulta aos resultados da Etapa 3

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parents[2]

ARQUIVO_PREVISOES = (
    BASE_DIR
    / "data"
    / "processed"
    / "previsoes_inadimplencia.csv"
)


def consultar_resultados() -> dict:
    """
    Consulta o CSV de previsões já existente e
    calcula indicadores para o agente de IA.
    """

    if not ARQUIVO_PREVISOES.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_PREVISOES}"
        )

    df = pd.read_csv(ARQUIVO_PREVISOES)

    colunas_obrigatorias = [
        "Inadimplencia_Real",
        "Probabilidade_Inadimplencia",
        "Inadimplencia_Prevista",
    ]

    colunas_ausentes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes: {colunas_ausentes}"
        )

    if df.empty:
        raise ValueError("O CSV não contém registros.")

    y_real = df["Inadimplencia_Real"]
    y_previsto = df["Inadimplencia_Prevista"]
    probabilidades = df["Probabilidade_Inadimplencia"]

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        y_previsto,
        labels=[0, 1]
    ).ravel()

    total = len(df)

    indicadores = {
        "total_registros": total,
        "total_sinalizados": int(y_previsto.sum()),
        "total_nao_sinalizados": int(
            (y_previsto == 0).sum()
        ),
        "percentual_sinalizados": round(
            float(y_previsto.mean() * 100), 2
        ),
        "probabilidade_media_percentual": round(
            float(probabilidades.mean() * 100), 2
        ),
        "inadimplentes_reais": int(y_real.sum()),
        "verdadeiros_positivos": int(tp),
        "verdadeiros_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "acuracia_percentual": round(
            accuracy_score(y_real, y_previsto) * 100, 2
        ),
        "precision_percentual": round(
            precision_score(
                y_real, y_previsto, zero_division=0
            ) * 100, 2
        ),
        "recall_percentual": round(
            recall_score(
                y_real, y_previsto, zero_division=0
            ) * 100, 2
        ),
        "f1_percentual": round(
            f1_score(
                y_real, y_previsto, zero_division=0
            ) * 100, 2
        ),
        "roc_auc_percentual": round(
            roc_auc_score(
                y_real, probabilidades
            ) * 100, 2
        ),
    }

    return indicadores


if __name__ == "__main__":
    resultado = consultar_resultados()

    print("=== INDICADORES DA ETAPA 3 ===")
    print(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        )
    )
