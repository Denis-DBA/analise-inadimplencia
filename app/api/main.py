
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

MODELO_PATH = BASE_DIR / "models" / "random_forest.joblib"
LIMIAR_PATH = BASE_DIR / "models" / "limiar.txt"

# Modelo já treinado na Etapa 2
modelo = joblib.load(MODELO_PATH)

# Limiar escolhido durante a validação
limiar = float(
    LIMIAR_PATH.read_text(encoding="utf-8").strip()
)

if not 0 <= limiar <= 1:
    raise ValueError("Limiar deve estar entre 0 e 1.")

app = FastAPI(
    title="API de Análise de Inadimplência",
    description="Previsões de inadimplência com Random Forest.",
    version="1.0.0",
)


class DadosCliente(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    X1: float = Field(ge=0, description="Limite de crédito")
    X2: int = Field(ge=1, le=2, description="Sexo codificado")
    X3: int = Field(description="Escolaridade codificada")
    X4: int = Field(description="Estado civil codificado")
    X5: int = Field(ge=18, le=120, description="Idade")

    X6: int
    X7: int
    X8: int
    X9: int
    X10: int
    X11: int

    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float

    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float

    @field_validator("X3")
    @classmethod
    def normalizar_escolaridade(cls, value):
        if value not in range(7):
            raise ValueError("Escolaridade fora do domínio documentado")
        return 4 if value in (0, 5, 6) else value

    @field_validator("X4")
    @classmethod
    def normalizar_estado_civil(cls, value):
        if value not in range(4):
            raise ValueError("Estado civil fora do domínio documentado")
        return 3 if value == 0 else value


@app.get("/")
def inicio():
    return {
        "projeto": "Análise de Inadimplência",
        "modelo": "Random Forest",
        "status": "API funcionando",
    }


@app.get("/health")
def verificar_saude():
    return {
        "status": "ok",
        "modelo_carregado": True,
        "limiar": limiar,
    }


class ResultadoPrevisao(BaseModel):
    probabilidade_inadimplencia: float = Field(ge=0, le=1)
    inadimplencia_prevista: int = Field(ge=0, le=1)
    classificacao: str
    limiar_utilizado: float = Field(ge=0, le=1)


@app.post("/predict", response_model=ResultadoPrevisao)
def prever_inadimplencia(cliente: DadosCliente):

    # Converter os 23 atributos para DataFrame
    dados = pd.DataFrame([cliente.model_dump()])

    # Reutilizar o pipeline treinado
    try:
        probabilidade = float(modelo.predict_proba(dados)[0, 1])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Atributos incompatíveis com o modelo.") from exc

    # Aplicar o mesmo limiar de validação
    previsao = int(probabilidade >= limiar)

    classificacao = (
        "Sinalizado"
        if previsao == 1
        else "Não sinalizado"
    )

    return {
        "probabilidade_inadimplencia": round(
            probabilidade, 6
        ),
        "inadimplencia_prevista": previsao,
        "classificacao": classificacao,
        "limiar_utilizado": limiar,
    }
