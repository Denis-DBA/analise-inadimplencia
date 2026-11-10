
# ETAPA 7 - PASSO 4
# Testes automatizados do Random Forest

from pathlib import Path

import joblib
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parents[1]

MODELO_PATH = BASE_DIR / "models" / "random_forest.joblib"
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset_teste.csv"
LIMIAR_PATH = BASE_DIR / "models" / "limiar.txt"


@pytest.fixture(scope="module")
def recursos():
    modelo = joblib.load(MODELO_PATH)
    df = pd.read_csv(DATASET_PATH)

    colunas = [f"X{i}" for i in range(1, 24)]
    dados = df.loc[[0], colunas]

    return modelo, dados


def test_modelo_carregado(recursos):
    modelo, _ = recursos

    assert hasattr(modelo, "predict_proba")


def test_quantidade_variaveis(recursos):
    _, dados = recursos

    assert dados.shape == (1, 23)


def test_probabilidade_valida(recursos):
    modelo, dados = recursos

    probabilidade = float(
        modelo.predict_proba(dados)[0, 1]
    )

    assert 0 <= probabilidade <= 1


def test_previsao_registro_conhecido(recursos):
    modelo, dados = recursos

    probabilidade = float(
        modelo.predict_proba(dados)[0, 1]
    )

    assert probabilidade == pytest.approx(
        0.141850310,
        abs=0.000001
    )


def test_limiar_classificacao(recursos):
    modelo, dados = recursos

    limiar = float(
        LIMIAR_PATH.read_text(encoding="utf-8").strip()
    )

    probabilidade = float(
        modelo.predict_proba(dados)[0, 1]
    )

    previsao = int(probabilidade >= limiar)

    assert limiar == pytest.approx(0.30)
    assert previsao == 0
