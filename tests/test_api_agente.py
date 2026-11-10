import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.api.main import app, modelo
from app.agents import agente_inadimplencia as agent

@pytest.fixture
def client():
    if os.getenv("AUDIT_NO_SOCKETS") == "1":
        pytest.skip("Sandbox bloqueia socketpair usado pelo TestClient no Windows")
    return TestClient(app)

@pytest.fixture
def payload():
    return pd.read_csv('data/processed/dataset_teste.csv').loc[0, [f'X{i}' for i in range(1,24)]].to_dict()

def test_health(client):
    assert client.get('/health').json()['modelo_carregado'] is True

def test_api_matches_model(client, payload):
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    p = modelo.predict_proba(pd.DataFrame([payload]))[0,1]
    assert response.json()['probabilidade_inadimplencia'] == pytest.approx(p, abs=1e-6)
    assert response.json()['inadimplencia_prevista'] == int(p >= .3)

@pytest.mark.parametrize('change', ['missing','extra','invalid','nonfinite'])
def test_invalid_payload(client, payload, change):
    if change=='missing': payload.pop('X1')
    elif change=='extra': payload['senha']='dummy'
    elif change=='invalid': payload['X1']='invalid'
    else: payload['X1']='Infinity'
    assert client.post('/predict',json=payload).status_code==422

def test_graph_historical(monkeypatch):
    monkeypatch.setattr(agent.llm.__class__, 'invoke', lambda *a, **k: type('Message', (), {'content':'Explicação simulada'})())
    result=agent.agente.invoke({'mensagem_usuario':'Explique os resultados'})
    assert result['indicadores']['total_registros']==6000
    assert result.get('resultado_ml') is None
    assert result['analise_ia']=='Explicação simulada'

def test_graph_prediction(monkeypatch, client, payload):
    monkeypatch.setattr(agent.requests,'post',lambda url, json, timeout: client.post('/predict',json=json))
    monkeypatch.setattr(agent.llm.__class__, 'invoke', lambda *a, **k: type('Message', (), {'content':'Explicação simulada'})())
    result=agent.agente.invoke({'mensagem_usuario':'Explique', 'dados_cliente':payload})
    assert result['resultado_ml']['inadimplencia_prevista']==0
    assert result.get('indicadores') is None

def test_api_failure_propagates(monkeypatch, payload):
    def fail(*args, **kwargs):
        raise agent.requests.Timeout('simulated')
    monkeypatch.setattr(agent.requests,'post', fail)
    with pytest.raises(agent.requests.Timeout):
        agent.consultar_modelo({'dados_cliente':payload})

def test_historical_clears_old_prediction():
    result=agent.consultar_resultados({'resultado_ml': {'old':True}})
    assert result['resultado_ml'] is None


def test_direct_prediction_and_schema(payload):
    from app.api.main import DadosCliente, ResultadoPrevisao, prever_inadimplencia
    response=ResultadoPrevisao.model_validate(prever_inadimplencia(DadosCliente.model_validate(payload)))
    expected=modelo.predict_proba(pd.DataFrame([payload]))[0,1]
    assert response.probabilidade_inadimplencia==pytest.approx(expected,abs=1e-6)

@pytest.mark.parametrize('field,value', [('X1','Infinity'),('X1','invalid'),('X2',1.5)])
def test_schema_rejects_bad_values(payload,field,value):
    from pydantic import ValidationError
    from app.api.main import DadosCliente
    payload[field]=value
    with pytest.raises(ValidationError): DadosCliente.model_validate(payload)

def test_group_split_prevents_overlap():
    from app.data_split import dividir_dados
    data=pd.read_csv('data/processed/dataset_limpo.csv')
    train,test,_,_=dividir_dados(data.drop(columns='Y'),data.Y)
    train_hash=pd.util.hash_pandas_object(train,index=False)
    test_hash=pd.util.hash_pandas_object(test,index=False)
    assert set(train_hash).isdisjoint(set(test_hash))
    assert len(train)+len(test)==30000
