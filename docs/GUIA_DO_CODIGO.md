# Guia do código: entendendo cada parte

Roteiro para ler o projeto na ordem em que os dados percorrem o sistema. Em cada parte: **o que faz, por que foi feito assim e onde olhar.**

```
PubMedQA + dados sintéticos do hospital
        │  anonimização → curadoria → split
        ▼
   train / validation / test (.jsonl)
        │  tokenização com máscara de prompt
        ▼
   QLoRA no Llama 3.2 (Colab, checkpoints no Drive)  ──►  adapter LoRA
        │                                                     │
        ▼                                                     ▼
   avaliação base × fine-tuned                      assistente (LangChain + LangGraph)
                                                    protocolos (RAG) + prontuário (SQLite)
                                                    guardrails + auditoria + validação humana
```

---

## 1. Dados (`src/data/`)

### PubMedQA (`pubmedqa.py`)
Perguntas biomédicas em inglês, cada uma com um resumo científico (`CONTEXTS`), a conclusão dos autores (`LONG_ANSWER`) e uma
decisão `yes/no/maybe`. O código transforma cada registro em uma conversa de chat:

- **user:** os contextos + a pergunta + a instrução de terminar com `Decision: yes|no|maybe`;
- **assistant:** a conclusão dos autores + `Decision: ...`.

Terminar com `Decision:` é o que permite **medir acerto** depois (extraímos a decisão por regex). O split (500 treino / 40 validação / 200 teste)
é embaralhado com semente fixa, então é reprodutível e não há vazamento entre os conjuntos.

### Dados do hospital (`synthetic_hospital.py` e `src/knowledge/`)
Dados reais de hospital são sigilosos (LGPD), então o projeto gera dados **sintéticos no mesmo formato** que dados reais teriam.
São 11 protocolos fictícios (sepse, dor torácica, AVC, pneumonia...), 6 pacientes e modelos de laudo/receita. Os exemplos de treino ensinam
**comportamentos**, não medicina:

1. responder só com base nos trechos fornecidos;
2. citar a fonte no formato `[ID#seção]`;
3. recusar pedidos de prescrição e perguntas sem base;
4. lembrar sempre a validação humana.

Alguns protocolos ficam **só no teste** (`HOLDOUT_PROTOCOLS`): é assim que se mede se o modelo generaliza em vez de decorar.

### Anonimização (`anonymization.py`)
Substitui por marcadores: `[CPF]`, `[TELEFONE]`, `[EMAIL]`, `[DATA]`, `[CRM]`, `[PRONTUARIO]`, `[NOME]`...
Roda `python scripts/demo_anonymization.py` para ver o antes e o depois com um texto fictício:

```
ANTES:  Paciente Maria Aparecida da Silva, prontuário 123456, CPF 123.456.789-09, nascida em 14/03/1958 ...
DEPOIS: Paciente [NOME], [PRONTUARIO], CPF [CPF], nascida em [DATA] ...
```

**Por que `anonymization_replacements` está vazio no `stats.json`?** A etapa roda sobre todos os exemplos do hospital, mas os dados sintéticos
já nascem sem dados pessoais, então não há o que substituir. Por isso existe o script de demonstração.
**Limite importante:** nomes só são detectados depois de um título (`Sr.`, `Dra.`, `Paciente`). Um nome solto no meio da frase passa.
Em produção seria necessário NER em português e revisão humana.

### Curadoria (`build_dataset.py`, função `curate`)
Remove duplicados e respostas vazias e, para o hospital, **descarta exemplos cuja resposta-alvo não passa nos guardrails de produção**
(dose, linguagem prescritiva, sem fonte). Assim o modelo só aprende comportamento seguro.

---

## 2. Tokenização e máscara de prompt (`src/finetuning/tokenization.py`)

O modelo recebe prompt + resposta, mas só deve **aprender a resposta**. Os tokens do prompt recebem `label = -100`, valor que o PyTorch ignora no cálculo da loss.
Sem isso, o modelo gastaria capacidade aprendendo a repetir perguntas e contextos.

Detalhes que importam:

- usa o `apply_chat_template` do próprio tokenizer, então o formato é o do Llama;
- exemplos acima de `max_length` (1536 tokens) são **descartados**, não cortados (cortar a resposta no meio ensinaria o modelo a não terminar frases);
- `tests/test_tokenization.py` testa isso com um tokenizer de mentira, sem baixar o modelo.

---

## 3. Fine-tuning QLoRA (`src/finetuning/train.py`)

**Conceitos**

- **LoRA:** em vez de atualizar bilhões de pesos, congela o modelo e treina pequenas matrizes extras (cerca de 1% dos parâmetros). O resultado é um *adapter* de poucas dezenas de MB.
- **QLoRA:** o modelo base fica quantizado em **4 bits** (NF4), o que faz o 3B caber na T4 de 16 GB; só o adapter é treinado, em precisão maior.

**Parâmetros e o porquê**

| Parâmetro | Valor | Motivo |
|---|---|---|
| 4 bits NF4 + dupla quantização | — | economiza memória sem perda grande de qualidade |
| `fp16=True`, `compute_dtype=float16` | — | **a T4 e a GTX 1660 Ti não têm `bf16`** |
| LoRA r=16, alpha=32, dropout 0.05 | 7 módulos | `q,k,v,o,gate,up,down`: cobre atenção e MLP |
| batch 1 × grad-accum 8 | efetivo 8 | cabe na memória e mantém estabilidade |
| lr 2e-4, cosseno, warmup 5% | — | valor típico de QLoRA |
| `max_grad_norm 0.3` | — | evita passos explosivos |
| `paged_adamw_8bit` | — | otimizador que usa menos memória |
| gradient checkpointing | — | troca tempo de cálculo por memória |
| 2 épocas | ~180 passos | com ~725 exemplos, mais passadas tendem a decorar |

**Checkpoints e retomada.** `save_steps=25` grava o estado completo no Drive. Ao iniciar, `get_last_checkpoint()` procura o mais recente e `trainer.train(resume_from_checkpoint=...)` continua dele.
Esse é o mecanismo que protege você das quedas do Colab gratuito.

**Resultado.** `final_adapter/` (só o adapter) e `training_summary.json` (parâmetros e histórico de loss).

---

## 4. Avaliação (`src/evaluation/`)

`evaluate.py` carrega o modelo base **uma vez**, aplica o adapter, gera as respostas do fine-tuned e depois usa `disable_adapter()` para gerar as do base, sem recarregar.
Os dois recebem os mesmos prompts, com decodificação gulosa (sem aleatoriedade), para a comparação ser justa.

| Grupo | Métricas (`metrics.py`) |
|---|---|
| PubMedQA | `decision_correct` (acerto de yes/no/maybe), `decision_missing`, `rouge_l` (similaridade com a conclusão dos autores) |
| Hospital | `citation_recall/precision`, `hallucinated_citation` (citou fonte que não estava no contexto), `dose` (deve ser 0), `disclaimer`, `refusal`, `no_base` |

Hipótese razoável para o relatório: o fine-tuning melhora principalmente **formato e comportamento** (citar fonte, recusar, não dar dose), e muito menos a precisão de `yes/no/maybe` do PubMedQA.
Só os números da sua execução dizem se isso se confirmou.

---

## 5. Assistente (`src/assistant/`)

- **`llm.py`:** três backends intercambiáveis: `fake` (para testes, sem GPU), `hf` (Llama + adapter) e `ollama`. Todos viram um *Runnable* do LangChain.
- **`retrieval.py`:** busca os trechos de protocolo mais parecidos com a pergunta usando **TF-IDF**. Escolha deliberada: roda offline, é determinístico e fácil de explicar. Para uma base maior, troque por embeddings.
- **`tools.py`:** três ferramentas LangChain (`get_patient_record`, `list_exams`, `search_protocols`) que dão acesso ao banco SQLite e aos protocolos.
- **`chains.py`:** pipeline LCEL: monta o prompt (paciente + trechos + pergunta) → LLM → texto. **O prompt é o mesmo do treino** (`src/prompts.py`), o que é essencial: o modelo foi ensinado a responder àquele formato.

### O grafo LangGraph (`graph.py`)

```
intake ─(bloqueado)───────────────────────┐
  │                                       │
load_patient ─(não encontrado)────────────┤
  │                                       │
check_pending_exams                       │
  │                                       │
retrieve_protocols                        │
  │                                       │
draft_answer (LLM)                        │
  │                                       │
safety_check ─(violação)─> fallback_answer┤
  │ (ok)                                  │
  └──────────────────────────────────> finalize ─(há alertas)─> emit_alerts ─> fim
```

Por que LangGraph e não só uma chain? Porque há **decisões no meio do caminho** (bloquear, paciente inexistente, reprovar no guardrail, emitir alertas) e cada nó é auditável e testável separadamente.
O diagrama em `docs/diagrama_fluxo.mmd` é gerado pelo próprio grafo (`python scripts/export_graph.py`) e renderiza direto no GitHub.

---

## 6. Segurança (`src/safety/`)

- **Guardrails de entrada:** detecta *prompt injection* ("ignore as instruções...") e bloqueia; marca pedidos de prescrição (`PEDIDO_DE_PRESCRICAO`).
- **Guardrails de saída:** rejeita resposta com **dose** (ex.: "500 mg"), **verbo prescritivo** ("administre", "prescrevo"), **sem fonte** ou com **fonte que não estava no contexto**.
  Se o rascunho da LLM for reprovado, entra o `fallback_answer`: uma resposta montada de forma determinística a partir dos protocolos.
- **Por que regex e não outra LLM?** São regras auditáveis, previsíveis e não dependem do modelo que estamos tentando controlar.
- **Auditoria (`logging_config.py`):** cada etapa grava um evento em `logs/audit.jsonl` com o mesmo `request_id`. Dá para reconstruir: pergunta → paciente → exames → fontes → resposta → guardrails → decisão do médico.
  Texto livre passa pelo anonimizador antes de ser gravado.
- **Explicabilidade (`explainability.py`):** cada resposta vem com as fontes citadas (e o trecho original), os campos do prontuário usados e um alerta se houver citação inventada.
- **Validação humana:** o estado final é sempre `PENDENTE_VALIDACAO_HUMANA`; o médico aprova ou rejeita e isso fica registrado.

---

## 7. Por que não continuar a tentativa com MLX do repositório do Gabriel

- O **MLX só roda em Mac com Apple Silicon**: não funciona no Colab nem em PCs com GPU NVIDIA. O caminho do modelo apontava para uma pasta do LM Studio no Mac.
- O notebook em Hugging Face estava incompleto: carregava o TinyLlama e o dataset, mas não tinha tokenização, `get_peft_model` nem `train()`. Além disso, usava `bfloat16`, que a T4 não suporta.
- Este projeto usa a pilha Hugging Face (`transformers` + `peft` + `bitsandbytes`), que roda no Colab e em GPUs NVIDIA.

---

## 8. Perguntas que podem surgir na reunião

**Por que Llama 3.2 3B e não 8B?** O 3B cabe com folga na T4 com QLoRA; o 8B fica no limite da memória e leva muito mais tempo. Dá para trocar mudando só `MODEL` no notebook.

**Por que treinar no Colab e não na GTX 1660 Ti?** Ela tem 6 GB de VRAM, que ficam no limite para o 3B em treino (só o 1B cabe com folga), e também não tem `bf16`. O Colab oferece ~16 GB. A máquina local roda o resto: dados, testes e o assistente.

**O fine-tuning ensina medicina ao modelo?** Não. Com poucos dados sintéticos, ele aprende **comportamento** (citar fonte, recusar prescrição, admitir falta de base). O conhecimento vem dos protocolos recuperados (RAG).

**Como garantimos que ele não prescreve?** Em camadas: o prompt, o treino com exemplos de recusa, o guardrail de saída (regex) e a validação humana obrigatória. A camada determinística vale mesmo se o modelo errar.

**E se o Colab cair no meio do treino?** O checkpoint a cada 25 passos no Drive permite retomar de onde parou.

**O que ainda pode dar errado?** O treino em GPU ainda não foi executado. O smoke test existe para descobrir isso cedo: tipicamente licença/token do Hugging Face, memória (OOM) ou versão de biblioteca.
