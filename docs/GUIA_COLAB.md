# Guia do Colab: treinar o modelo passo a passo

Objetivo: fazer o fine-tuning do Llama 3.2 no Colab gratuito, **sem perder o progresso** se a sessão cair.

## Antes de abrir o Colab (faça uma vez)

1. **Push do projeto para o GitHub.** O notebook 02 faz `git clone` do seu repositório, então ele precisa estar lá.
   Comandos na seção 6 do `README.md`.
2. **Licença do Llama.** Entre em <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>, faça login e aceite os termos.
   A aprovação pode demorar; faça isso agora. (Opcional: o mesmo para o `Llama-3.2-1B-Instruct`, usado nos testes rápidos.)
3. **Token.** Em <https://huggingface.co/settings/tokens> crie um token do tipo *Read*.
4. **Secret no Colab.** No Colab, ícone de chave (barra lateral) → *Adicionar secret* → nome `HF_TOKEN`, valor = seu token,
   e ative **Acesso pelo notebook**.

## Executando

1. Abra o notebook: Colab → *Arquivo → Abrir notebook → GitHub* → cole `nsouzajrs/desafio-tecnico-fase3` → `notebooks/02_fine_tuning_colab.ipynb`.
2. *Ambiente de execução → Alterar tipo de ambiente de execução → **T4 GPU***.
3. Execute as células **em ordem**. O que cada bloco faz:

| Bloco | O que acontece | O que conferir |
|---|---|---|
| 1. GPU | roda `nvidia-smi` | aparece "Tesla T4". O treino usa `fp16` de qualquer forma (a T4 não tem `bf16` nativo) |
| 2. Setup | monta o Drive, clona o repo, instala dependências | sem erros de `pip` |
| 3. Token | lê o secret `HF_TOKEN` | `Token carregado: True` |
| 4. Configuração | define `MODEL` e `RUN_NAME` (pasta no Drive) | `Checkpoints em: /content/drive/MyDrive/fase3/...` |
| 5. Dataset | gera `train/validation/test.jsonl` | contagens parecidas com `data/processed/stats.json` |
| 6. **Smoke test** | treina com 32 exemplos por poucos passos | termina sem erro e salva um adapter |
| 7. Treino completo | QLoRA, 2 épocas | a loss de treino cai; aparecem linhas `eval_loss` |
| 8. Curva | gráfico treino × validação | veja a interpretação abaixo |
| 9. Adapter final | lista `final_adapter` no Drive | arquivos `adapter_model.safetensors` e `adapter_config.json` |

**Não pule o smoke test.** Deve levar poucos minutos (estimativa, ainda não medida) e descobre problemas de licença, memória ou versão de biblioteca
antes de você gastar uma hora de GPU.

## Como funcionam os checkpoints

- O treino grava um checkpoint a cada **25 passos** em `MyDrive/fase3/<RUN_NAME>/checkpoint-XX`
  (estado do adapter + otimizador + agendador). Ficam só os 2 mais recentes.
- Se a sessão cair: reabra o notebook, rode as células 1 a 4 (GPU, setup, token, configuração) e depois a célula do treino completo,
  **com o mesmo `RUN_NAME`**. A linha `>>> Retomando do checkpoint: .../checkpoint-XX` confirma que continuou de onde parou.
- Para começar um experimento novo sem apagar o anterior, troque `RUN_NAME` (ex.: `run2_3b_lr1e-4`).

### Quanto tempo leva?

Com ~725 exemplos e batch efetivo 8, são cerca de **91 passos por época, ~180 no total** (2 épocas). O tempo por passo na T4
**ainda não foi medido**: veja a barra de progresso nos primeiros passos e multiplique. Se estiver lento demais, use o 1B
(`meta-llama/Llama-3.2-1B-Instruct`) ou reduza `--max-length` e os exemplos de PubMedQA (`--max-pubmed-train` no passo 5).

## Lendo a curva de aprendizado

- **Loss de treino caindo e loss de validação caindo:** ótimo.
- **Treino cai, validação sobe:** *overfitting* (decorou os exemplos). Reduza para 1 época ou use menos passos.
- **Nada cai:** confira a taxa de aprendizado e se o dataset foi gerado (célula 5).

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `401` / `403` / "gated repo" ao baixar o modelo | licença não aceita ou token errado | aceite a licença no Hugging Face; confira o secret `HF_TOKEN` e o acesso pelo notebook |
| `CUDA out of memory` | sequência longa ou modelo grande | `--max-length 1024`, ou use o modelo 1B; reinicie o ambiente de execução antes de tentar de novo |
| "GPU não encontrada" | ambiente sem T4 | *Alterar tipo de ambiente de execução → T4 GPU* |
| Sessão desconectou / cota de GPU acabou | limite do Colab gratuito | espere a cota voltar, rode as células 1–4 e a de treino; ele retoma do checkpoint |
| Erro com `bitsandbytes` | versão incompatível | reinicie o ambiente de execução e rode o setup de novo |
| Drive não monta | autorização pendente | rode a célula de novo e aceite a janela de permissão |

## Levando o resultado para o seu computador

1. No Drive, baixe a pasta `fase3/<RUN_NAME>/final_adapter` (poucas dezenas de MB).
2. Salve em `artifacts/adapter/` dentro do projeto.
3. Use com `python app.py --backend hf --adapter artifacts/adapter ...` (README, seção 5).

Guarde também `training_summary.json` (curva de loss e parâmetros): ele alimenta o relatório técnico.
