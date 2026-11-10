
# ETAPA 7 - PASSO 3
# Testes automatizados dos indicadores

from app.agents.consultar_resultados import consultar_resultados
import sys
from pathlib import Path

# Localizar a raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_total_registros():
    indicadores = consultar_resultados()

    assert indicadores["total_registros"] == 6000


def test_total_sinalizados():
    indicadores = consultar_resultados()

    assert indicadores["total_sinalizados"] == 1380


def test_matriz_confusao():
    indicadores = consultar_resultados()

    assert indicadores["verdadeiros_positivos"] == 737
    assert indicadores["verdadeiros_negativos"] == 4030
    assert indicadores["falsos_positivos"] == 643
    assert indicadores["falsos_negativos"] == 590


def test_metricas():
    indicadores = consultar_resultados()

    assert indicadores["acuracia_percentual"] == 79.45
    assert indicadores["precision_percentual"] == 53.41
    assert indicadores["recall_percentual"] == 55.54
    assert indicadores["f1_percentual"] == 54.45
    assert indicadores["roc_auc_percentual"] == 78.42
