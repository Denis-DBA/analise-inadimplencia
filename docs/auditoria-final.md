# Auditoria final — 10/11/2026

## Resultado e localização

A versão preparada está na cópia de trabalho desta sessão, separada da pasta original. Não foi possível modificar diretamente a pasta original devido às permissões do ambiente. Os dados e modelos originais não foram excluídos nem sobrescritos. Nenhuma credencial foi alterada ou revogada. Não houve publicação, push nem alteração de visibilidade no GitHub.

## Segurança

O inventário recursivo, incluindo itens ocultos, encontrou 33.379 arquivos na pasta original: 33.335 pertenciam ao ambiente virtual. Os 44 arquivos restantes incluíam 26 arquivos de texto analisados por padrões de credenciais e 18 binários. Não houve erro de leitura nem correspondência aos padrões de chaves privadas, tokens GitHub/AWS/serviços, credenciais em URLs, atribuições de segredo ou e-mail nesses textos.

Isso não constitui garantia de ausência de qualquer segredo. O ambiente virtual foi identificado e excluído da distribuição; seu código de terceiros não foi submetido à mesma varredura textual. Binários, estado local do LangGraph e caches foram excluídos da publicação. O pipeline Joblib foi carregado para validação, e os CSVs e a planilha foram examinados para estrutura e consistência; esses dados não serão publicados.

Não havia `.git` na raiz original nem nos diretórios ancestrais verificados. O comando Git diretamente na pasta original foi bloqueado por permissão; não existia histórico local disponível para auditoria. Um novo repositório local foi preparado somente com os arquivos selecionados da cópia de trabalho. A busca no GitHub conectado não encontrou repositório `inadimplencia` da conta Denis-DBA. Nenhum remoto foi criado ou sobrescrito.

Proteções adicionadas: exclusão de dados preparados/brutos, relatórios, modelos Joblib/Pickle, caches e arquivos temporários; remoção de todas as saídas dos notebooks; `.dockerignore` para reduzir o contexto de build; identidade de commit local com endereço GitHub noreply.

## Problemas técnicos corrigidos

- A divisão original permitia 16 correspondências de atributos idênticos entre treino e teste. A divisão e os folds de validação passaram a agrupar atributos idênticos, preservando as 30.000 linhas. A verificação final encontrou zero grupos compartilhados entre treino e teste.
- Regressão Logística e Random Forest foram retreinados; a comparação e a análise de limiares foram reexecutadas. O limiar permanece uma escolha explícita de 0,30.
- API passou a rejeitar valores não finitos, validar parte dos domínios, normalizar categorias como no treinamento e validar a saída.
- Cada rota do agente limpa o resultado da rota anterior, evitando analisar um resultado antigo presente no estado.
- Endereços Docker foram retirados da configuração local e movidos para o Compose. Foram removidos nomes fixos dos containers.
- Dependências diretas foram fixadas nas versões instaladas e utilizadas na validação. Os Dockerfiles passaram a compartilhar esse arquivo.
- A ingestão passa a gravar uma cópia bruta na primeira execução, sem sobrescrever uma cópia existente; essa alteração ainda requer validação com acesso à UCI.

## Validações executadas

| Verificação | Resultado |
|---|---|
| Suíte original antes das alterações | 9 testes aprovados |
| Suíte final | 17 aprovados; 7 ignorados por bloqueio de sockets |
| Avisos na suíte | 13 avisos de depreciação de dependências |
| Notebook 01 | Tentativa de execução interrompida na obtenção UCI por falha de conexão |
| Notebook 02 | Células de código executadas, treinamento e validação cruzada concluídos |
| Notebook 03 | Células computacionais executadas, avaliação, permutação e exportações concluídas |
| Instalação embutida em notebook | Não repetida; pacotes já instalados |
| Quatro notebooks | JSON, esquema nbformat e sintaxe das células validados; saídas removidas |
| Dataset preparado | 30.000 linhas, 24 colunas, zero valores ausentes |
| Separação final | 24.000 treino, 6.000 teste, sem grupos compartilhados |
| Modelo persistido | Previsões dos 6.000 registros coincidem com CSV exportado |
| Excel | Duas abas geradas; primeira aba com 6.000 previsões |
| API | Esquemas e inferência direta executados; equivalência com pipeline confirmada |
| LangGraph | Rota histórica executada com Ollama simulado; testes de limpeza de estado e falha de rede aprovados |
| Compose | `docker compose -f docker/compose.yaml config --quiet` retornou sucesso |

As células computacionais de 02/03 foram executadas por IPython em processo separado, sem interface gráfica, usando o dataset preparado copiado do original. Não foi executado um kernel Jupyter via nbclient; os gráficos utilizaram backend não interativo. Os dados foram regenerados apenas na cópia de trabalho.

## Resultados do modelo corrigido

| Métrica | Valor |
|---|---:|
| Acurácia | 79,45% |
| Precision | 53,41% |
| Recall | 55,54% |
| F1 | 54,45% |
| ROC-AUC | 78,42% |
| TN / FP / FN / TP | 4030 / 643 / 590 / 737 |

Os valores anteriores (F1 54,09%, ROC-AUC 77,62%) foram reproduzidos antes da correção e não descrevem o modelo atual. Evidências agregadas e hashes dos artefatos locais estão em `metricas-verificadas.json`.

## Limitações e pendências reais

- A fonte oficial não foi baixada nesta sessão; a preservação dos 30.000 registros foi confirmada no dataset preparado existente, sem comparação célula a célula com uma nova cópia da UCI.
- Os 7 testes ignorados usam HTTP/TestClient. Uma tentativa real bloqueou no `socketpair` interno do asyncio no Windows e foi interrompida. O notebook 04, que depende de uma API HTTP em execução, não foi executado integralmente.
- A rota completa de nova previsão no grafo com HTTP simulado também depende de TestClient e ficou entre os testes ignorados. A função de previsão e o roteamento histórico foram testados separadamente.
- Ollama não estava acessível por HTTP; não foi validada uma resposta generativa real.
- Docker Engine retornou permissão negada: imagens não foram construídas e containers não foram iniciados.
- A instalação em ambiente limpo e a resolução de dependências transitivas não foram testadas.
- Não houve auditoria de vulnerabilidades de todas as dependências nem análise de equidade completa.

Essas limitações estão documentadas no README. Não existe serviço contínuo de ingestão implementado. PostgreSQL e n8n permanecem exclusivamente no roadmap.

## Publicação

Nome proposto: `Denis-DBA/analise-inadimplencia`. A criação pública e o push final dependem de confirmação explícita do usuário. A conexão GitHub foi confirmada, mas o GitHub CLI local não está autenticado. Nenhum comando de força, reescrita de histórico remoto ou exclusão remota foi realizado.
