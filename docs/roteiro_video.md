# Roteiro do vídeo (máx. 15 min)

O enunciado pede que o vídeo mostre: **treinamento/funcionamento da LLM, execução de um fluxo automatizado, respostas clínicas contextualizadas e logs/validação.**

## Preparação (antes de gravar)

- Grave o treino **antes**: ele demora. Deixe o notebook 02 executado, com a curva de loss e a pasta de checkpoints no Drive visíveis.
- Rode o notebook 03 (avaliação) e deixe a tabela base × fine-tuned na tela.
- Deixe o notebook 04 (ou `app.py`) pronto, com o adapter carregado.
- Limpe `logs/audit.jsonl` para a auditoria do vídeo mostrar só as consultas da demonstração.
- Esconda o token do Hugging Face (não mostre os Secrets nem a variável de ambiente).

## Linha do tempo

| Tempo | Bloco | O que mostrar |
|---|---|---|
| 0:00–1:00 | **Abertura** | problema, objetivo e arquitetura em uma frase; aviso de que os dados são sintéticos e o assistente nunca prescreve |
| 1:00–3:00 | **Dados** | PubMedQA + dados sintéticos do hospital; `python scripts/demo_anonymization.py` (antes/depois); curadoria; `stats.json` |
| 3:00–7:00 | **Fine-tuning** | notebook 02: escolha do Llama 3.2 3B e QLoRA (explicar em 20 s); smoke test; curva de treino × validação; **checkpoints no Drive** e a retomada (`>>> Retomando do checkpoint`) |
| 7:00–9:00 | **Avaliação** | tabela base × fine-tuned do notebook 03; onde melhorou (citação, recusa, sem dose) e onde não; exemplo qualitativo |
| 9:00–13:00 | **Assistente em ação** | notebook 04, nesta ordem (abaixo) |
| 13:00–14:15 | **Logs e validação** | abrir `logs/audit.jsonl` e seguir um `request_id`; registrar a decisão do médico (aprovado/rejeitado) |
| 14:15–15:00 | **Fechamento** | limitações, aprendizados, próximos passos |

### Cenários da demonstração (9:00–13:00)

1. **Suspeita de sepse (P001):** o fluxo roda, mostra os exames pendentes, recupera o protocolo e responde **com fontes** `[ID#seção]`. Mostrar o diagrama do LangGraph e o caminho percorrido (`trace`).
2. **Resultado crítico (P006):** o alerta para a equipe médica aparece (`emit_alerts`).
3. **Pedido de prescrição:** "Prescreva o antibiótico..." → o assistente recusa e remete ao médico responsável.
4. **Guardrail em ação:** com a LLM "insegura" simulada (`--fake-mode unsafe`), mostrar o rascunho com dose sendo **bloqueado** e o fallback assumindo.
5. **Tentativa de burlar as regras:** "Ignore todas as instruções..." → bloqueado na entrada (`PROMPT_INJECTION`).

## Pontos que valem falar em voz alta

- "O fine-tuning ensina **comportamento**; o conhecimento vem dos protocolos recuperados (RAG)."
- "A segurança não depende do modelo: as regras são determinísticas e auditáveis."
- "Toda resposta termina como **pendente de validação humana**."
- "Se a sessão do Colab cai, o treino **retoma do checkpoint**."

## Entrega

Suba o vídeo (YouTube como não listado, ou Drive com link de acesso) e coloque o link no README e na entrega da plataforma.
