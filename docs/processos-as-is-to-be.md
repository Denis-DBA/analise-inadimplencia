
# Modelagem de Processos — As-Is e To-Be

## 1. Objetivo

Documentar o processo de análise de inadimplência
utilizando Machine Learning, Inteligência Artificial
Generativa e automação.

O projeto utiliza dados públicos do dataset
Default of Credit Card Clients, disponibilizado
pela UCI Machine Learning Repository.

Os dados são históricos e utilizados para fins
educacionais e experimentais.

## 2. AS-IS — Processo Inicial

O processo inicial foi desenvolvido com execução
manual de notebooks Jupyter.

### Fluxo As-Is

1. Obter o dataset público da UCI.
2. Carregar os dados utilizando Python e Pandas.
3. Executar a preparação e limpeza dos dados.
4. Treinar o modelo Random Forest.
5. Avaliar as previsões no conjunto de teste.
6. Gerar indicadores estatísticos.
7. Exportar os resultados para CSV e Excel.
8. Interpretar os relatórios manualmente.

### Limitações

- Dependência de execução manual dos notebooks.
- Necessidade de conhecimento técnico para consultas.
- Ausência de orquestração automática entre serviços.
- Atualização de relatórios dependente de nova execução.

## 3. Estado Atual — Implementado

O projeto já possui:

- Pipeline de preparação dos dados.
- Random Forest treinado e salvo com Joblib.
- Avaliação com 6.000 registros de teste.
- Relatório Excel com previsões e indicadores.
- API FastAPI para novas previsões.
- Agente LangGraph com roteamento condicional.
- Integração local com Ollama e Qwen.
- Consultas aos indicadores históricos por linguagem natural.

## 4. TO-BE — Processo Automatizado Proposto

### Fluxo To-Be

1. Receber ou consultar dados estruturados.
2. Validar os dados de entrada.
3. Acionar a API de Machine Learning quando
   uma nova previsão for necessária.
4. Consultar indicadores históricos disponíveis.
5. Utilizar LangGraph para coordenar as operações.
6. Utilizar Qwen para gerar análises explicativas.
7. Automatizar consultas e relatórios com n8n.
8. Disponibilizar resultados em Excel.
9. Encaminhar resultados para revisão humana.

### Benefícios esperados

- Menor dependência de tarefas manuais.
- Padronização das consultas e respostas.
- Maior rastreabilidade do processamento.
- Integração entre Machine Learning e automação.
- Facilidade de consulta aos indicadores.

Os benefícios operacionais ainda precisam ser
medidos após a implementação do fluxo automatizado.

## 5. Tecnologias

- Python
- Pandas
- Scikit-learn
- Random Forest
- FastAPI
- LangGraph
- Ollama / Qwen
- n8n (integração planejada)
- Microsoft Excel

## 6. Considerações

O modelo utiliza dados históricos da UCI e não foi
validado para decisões reais de concessão de crédito.

As previsões são experimentais e devem ser
interpretadas como resultados estatísticos.

O processo To-Be representa a arquitetura proposta.
As funcionalidades ainda não implementadas não são
apresentadas como entregas concluídas.
