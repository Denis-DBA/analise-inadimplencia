# Análise de Inadimplência com Machine Learning e IA Generativa

Projeto de portfólio em Python que transforma dados históricos de crédito em um modelo de classificação, relatórios CSV/Excel, API REST e consultas explicadas por um agente LangGraph com Ollama.

O problema de negócio é identificar sinais estatísticos associados à inadimplência no mês seguinte e avaliar o compromisso entre detectar casos positivos e gerar falsos alertas. A solução é experimental e não está validada para decisões reais de crédito.

## Arquitetura implementada

1. O notebook 01 obtém o dataset oficial, explora a qualidade e normaliza categorias.
2. O notebook 02 compara Regressão Logística e Random Forest, valida os modelos e salva o pipeline Random Forest e o limiar.
3. O notebook 03 avalia o conjunto reservado e gera previsões, indicadores, gráficos e planilha Excel.
4. FastAPI carrega o pipeline persistido e recebe os atributos `X1` a `X23` em `/predict`.
5. LangGraph seleciona uma consulta ao CSV histórico ou uma previsão pela FastAPI. Ollama explica o resultado numérico recebido.

O código Python realiza ingestão pontual no notebook 01. Não existe monitoramento contínuo, simulador de novos registros ou serviço automático de ingestão nesta versão.

## Tecnologias e estrutura

Python 3.14.6, pandas, NumPy, scikit-learn, Joblib, Matplotlib, FastAPI, Pydantic, LangGraph, LangChain Ollama, Requests, Jupyter, OpenPyXL, Pytest e Docker Compose. As versões diretas do ambiente de validação estão em `requirements.txt`; esse arquivo não é um lock completo de dependências transitivas.

| Caminho | Conteúdo |
|---|---|
| `01_preparacao_limpeza.ipynb` | Ingestão UCI, exploração e normalização |
| `02_treinamento.ipynb` | Treinamento, comparação, validação e persistência |
| `03_avaliacao_resultados.ipynb` | Avaliação final e exportações |
| `04_testes_api.ipynb` | Comparação entre previsão direta e HTTP |
| `app/data_split.py` | Separação estratificada por grupos de atributos idênticos |
| `app/api/main.py` | API e validação dos atributos |
| `app/agents/` | Grafo e cálculo dos indicadores históricos |
| `data/raw/` | Cópia local obtida na ingestão, excluída do Git |
| `data/processed/` | Datasets e relatórios gerados, excluídos do Git |
| `models/` | Pipeline local e limiar; binários excluídos do Git |
| `docker/` | Dockerfiles e Compose |
| `tests/` | Regressão do modelo, indicadores, esquema, API e agente |
| `docs/` | Processo e relatório de auditoria |

## Fonte e preservação dos dados

[Default of Credit Card Clients — UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), de I-Cheng Yeh (2009), DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H). A fonte informa 30.000 registros, 23 atributos e licença CC BY 4.0. São dados históricos de Taiwan, com pagamentos de 2005; valores monetários são expressos em NT$.

O notebook usa `fetch_ucirepo(id=350)`. A normalização mapeia escolaridade `0/5/6` para `4` e estado civil `0` para `3`, mantendo todas as linhas. Duplicatas e valores extremos não são excluídos. A ingestão passa a salvar uma cópia em `data/raw/default_credit_clients.csv`, sem sobrescrever uma cópia já existente. A cópia preparada disponível na auditoria contém 30.000 linhas e 24 colunas, sem valores ausentes. A recuperação da fonte original não pôde ser reexecutada nesta sessão por bloqueio de rede; não foi possível comparar todas as células com uma nova cópia oficial.

Dados individuais, relatórios e modelos binários não acompanham o repositório. Recrie-os executando os notebooks. A licença do dataset é distinta da licença MIT do código.

## Pipeline e prevenção de vazamento

O conjunto reservado é definido pelo primeiro fold de `StratifiedGroupKFold`, com cinco folds, embaralhamento e `random_state=42`. A divisão validada resultou em 24.000 registros de treino e 6.000 de teste. Atributos idênticos recebem o mesmo grupo, mesmo quando o alvo diverge, evitando que cruzem essa fronteira. A validação cruzada de treinamento também respeita esses grupos.

O pipeline inclui `StandardScaler` para variáveis numéricas e `OneHotEncoder(handle_unknown="ignore")` para `X2`, `X3` e `X4`. As transformações são ajustadas dentro do pipeline e dos folds de treinamento. A correlação exploratória no notebook 01 usa o conjunto completo, mas não é usada para selecionar atributos ou hiperparâmetros.

Foram executados treinamento e validação cruzada de Regressão Logística e Random Forest. O Random Forest persistido tem 200 árvores, profundidade máxima 12, mínimo de cinco amostras por folha e semente 42. O limiar operacional é fixado em 0,30 depois da comparação de limiares em probabilidades fora do fold do treinamento; não há seleção automática desse valor. A comparação de modelos e limiares não usa o teste final.

A avaliação inclui métricas de classificação, matriz de confusão, curvas ROC e Precision-Recall, Brier Score e importância de atributos. A importância por permutação é calculada no treino e deve ser interpretada com essa limitação.

## Resultados verificados em 10/11/2026

Resultados do modelo retreinado após corrigir a separação de registros idênticos. Não confundir com os resultados da divisão aleatória anterior.

| Métrica | Resultado no teste |
|---|---:|
| Acurácia | 79,45% |
| Precision | 53,41% |
| Recall | 55,54% |
| F1-score | 54,45% |
| ROC-AUC | 78,42% |
| Limiar | 0,30 |

| Matriz de confusão | Quantidade |
|---|---:|
| Verdadeiros negativos | 4.030 |
| Falsos positivos | 643 |
| Falsos negativos | 590 |
| Verdadeiros positivos | 737 |

Esses números são de uma única divisão reproduzível. Não representam garantia de desempenho futuro. Os testes de regressão precisam ser revistos caso o dataset, as versões ou o procedimento de treinamento mudem.

## Instalação e reprodução

Na raiz do projeto, em PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Selecione o kernel desse ambiente e execute os notebooks 01, 02 e 03, nessa ordem, a partir da raiz. O notebook 01 requer internet para obter os dados. O treinamento e a importância por permutação podem levar alguns minutos. Os comandos de instalação embutidos em duas células são opcionais quando os requisitos já foram instalados.

Para servir o modelo gerado:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

O modelo e `models/limiar.txt` precisam existir antes de importar a aplicação. A inicialização falha se estiverem ausentes ou inválidos; `/health` não é uma sonda independente dos artefatos.

## API e exemplo de utilização

| Método | Rota | Finalidade |
|---|---|---|
| GET | `/` | Identificação da aplicação |
| GET | `/health` | Modelo carregado e limiar |
| POST | `/predict` | Probabilidade, classe e limiar |
| GET | `/docs` | Interface OpenAPI |

Exemplo sintético, sem dados privados, em Python:

```python
import requests

cliente = {f"X{i}": 0 for i in range(1, 24)}
cliente.update(X1=50000, X2=1, X3=2, X4=1, X5=35)
resposta = requests.post(
    "http://127.0.0.1:8000/predict", json=cliente, timeout=30
)
resposta.raise_for_status()
print(resposta.json())
```

A entrada rejeita atributos ausentes ou extras, valores não numéricos e infinitos. Sexo, idade, escolaridade e estado civil têm validações de domínio; os códigos normalizados na preparação também são normalizados na API. A saída é validada por um esquema Pydantic. Consulte a documentação UCI para o significado de cada atributo. Ainda não existe validação de domínio completa para todos os campos financeiros.

O notebook 04 requer a API HTTP em execução. A API não tem autenticação nesta versão; a configuração de execução e as portas do Compose ficam vinculadas ao endereço local.

## LangGraph e Ollama

Instale o Ollama separadamente e disponibilize o modelo:

```powershell
ollama pull qwen3.5:9b
.\.venv\Scripts\langgraph.exe dev
```

Por padrão, o agente consulta a API em `http://127.0.0.1:8000/predict` e o Ollama em `http://127.0.0.1:11434`. As variáveis `API_INADIMPLENCIA_URL`, `OLLAMA_BASE_URL` e `OLLAMA_MODEL` permitem alterar esses destinos. A configuração local não fixa endereços da rede Docker.

No Studio, envie `{"mensagem_usuario": "Explique as métricas"}` para consultar o relatório histórico. Para nova inferência, envie também `dados_cliente` com os 23 atributos do exemplo anterior. As duas rotas convergem para a explicação pelo modelo de linguagem, que recebe os números do sistema. As instruções do agente proíbem inventar métricas ou alterar probabilidades, mas essa orientação não garante que toda resposta generativa seja correta.

Falhas HTTP são propagadas após timeout de 30 segundos e `raise_for_status`; erros do Ollama também são propagados. Ainda não há retentativas nem resposta de recuperação padronizada.

## Docker

Gere os artefatos nos notebooks antes de construir as imagens:

```powershell
docker compose -f docker/compose.yaml config --quiet
docker compose -f docker/compose.yaml up --build
```

Compose sobe a API e o servidor de desenvolvimento LangGraph, aguarda a saúde da API e configura os endereços internos. O Ollama continua no host, acessível como `host.docker.internal`; precisa aceitar conexões dessa rede. As portas publicadas são locais: 8000 e 2024. Verifique conflitos com serviços existentes antes de executar.

Os modelos e relatórios são copiados para as imagens durante o build. Não há volume de persistência nem atualização automática desses arquivos; retreinar exige reconstruir a imagem. O contexto Docker exclui ambientes, caches e credenciais. O uso de `langgraph dev` é demonstrativo, sem caracterizar implantação de produção.

A configuração Compose foi validada sintaticamente nesta auditoria. O build e a execução dos containers não foram realizados porque o ambiente bloqueou o acesso ao Docker Engine.

## Testes e limites da validação

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Os testes dependem dos artefatos gerados. O relatório detalhado da sessão está em `docs/auditoria-final.md`. Testes com Ollama simulado validam o roteamento do grafo, não a qualidade de respostas reais. Testes HTTP com `TestClient` não puderam ser concluídos neste ambiente Windows por bloqueio de sockets; a validação direta de funções e esquemas foi executada separadamente. O Ollama real e o acesso à fonte UCI não estavam acessíveis.

Os notebooks publicados não contêm saídas armazenadas, tabelas individuais nem caminhos locais da máquina de desenvolvimento.

## Limitações e próximos passos

- Base histórica e geográfica restrita, com variáveis demográficas; não houve auditoria completa de equidade, drift ou validação temporal.
- Não há monitoramento contínuo, retreinamento automático, observabilidade ou controle de acesso implementados.
- Reexecutar a ingestão oficial, os testes HTTP, o Ollama real e o build Docker em um ambiente com os acessos necessários.
- PostgreSQL, futuramente: persistência e gerenciamento de dados.
- n8n, futuramente: automação e orquestração de workflows.

PostgreSQL e n8n não fazem parte da implementação desta publicação.

## Licença

Código sob licença MIT, conforme `LICENSE`. Dataset atribuído à UCI e a I-Cheng Yeh sob CC BY 4.0.
