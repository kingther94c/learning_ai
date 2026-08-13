# Learning path

This repository is a ladder, not an encyclopedia. Each rung should make one
mechanism small enough to print, prove it against a trusted reference, and then
let you break it on purpose.

## How to work through a lesson

Use four passes. They are more important than finishing quickly.

1. **Predict.** Before running a cell or moving a control, write down what you
   expect to change and what should stay invariant.
2. **Build.** Re-type the central operation once. Do not copy the whole notebook.
3. **Verify.** Read every `assert` and shape annotation. Explain why that is the
   right test rather than merely accepting the green check.
4. **Perturb.** Complete the `Your turn` cells, choose one assumption to violate,
   and record the first surprising result.

For a topic to count as learned, you should be able to explain its input/output
shapes, implement the smallest correct version, name one failure mode, and design
an experiment that reveals that failure.

## Curriculum

| Stage | Core questions | Build | Interactive intuition |
|---|---|---|---|
| 1. ML foundations | What can a model learn? When does it generalise? | linear/logistic regression, trees, k-means, PCA | loss surfaces, decision boundaries, bias–variance |
| 2. Deep learning | How does credit flow backward? Why does training become unstable? | autograd scalar engine, MLP, initialisation, optimisers, normalisation | computation graph, gradient flow, optimisation trajectories |
| 3. NLP | How does text become numbers? What did sequence models do before Transformers? | character/word/BPE tokenisers, embeddings, n-grams, RNN/LSTM | merge process, embedding geometry, vanishing memory |
| 4. Transformers | How can tokens exchange information in parallel? | attention, masking, multi-head attention, residual stream, MLP, positional encoding | Q/K/V routing, masks, position mechanisms |
| 5. GPT | How does next-token prediction become generation and in-context learning? | decoder-only LM, sampling, KV cache, scaling experiments, induction task | sampling distribution, cache growth, learned heads |
| 6. Frontier | Which newer ideas change routing, memory, alignment, or reasoning? | small phenomenon-first paper reproductions | routing load, state memory, preference margins, test-time compute |

Start with
[`04_transformer/01_build_a_transformer.ipynb`](../notebooks/04_transformer/01_build_a_transformer.ipynb).
It intentionally bridges stages 3–5; afterwards, fill earlier gaps or continue
into GPT depending on which part felt least solid.

Then use
[`04_transformer/02_build_a_bert.ipynb`](../notebooks/04_transformer/02_build_a_bert.ipynb)
as the controlled comparison: remove GPT's causal mask, corrupt targets with
masked language modelling, and see how one encoder serves token- and
sequence-level understanding tasks.

Three classic-paper microscopes can be studied independently or in historical order:

1. [`03_nlp/01_word2vec.ipynb`](../notebooks/03_nlp/01_word2vec.ipynb) — learn a
   static vector from local context before asking Transformers for contextual vectors.
2. [`02_deep_learning/01_resnet.ipynb`](../notebooks/02_deep_learning/01_resnet.ipynb) —
   understand the identity shortcut that later Transformer blocks inherit.
3. [`06_frontier/01_lora.ipynb`](../notebooks/06_frontier/01_lora.ipynb) — freeze a
   pretrained matrix and adapt it through a low-rank update.

## Planned lesson sequence

The next useful lessons are ordered by dependency, not hype:

1. `03_nlp/02_tokenization` — character, word and BPE tokenisation from scratch.
2. `02_deep_learning/02_backprop` — a scalar autograd engine and gradient checks.
3. `05_gpt/01_sampling` — temperature, top-k and top-p from raw logits.
4. `05_gpt/02_kv_cache` — prove equivalence, then measure the complexity change.
5. `05_gpt/03_induction_heads` — train a two-layer attention model on the
   `[A][B] … [A] → [B]` task and intervene on the responsible head.
6. `06_frontier/02_moe` — top-k routing, capacity, load balancing and expert
   collapse on a toy mixture.
7. `06_frontier/03_selective_ssm` — scan versus attention on selective-copy and
   long-memory tasks.
8. `06_frontier/04_dpo` — derive and implement the preference loss on a tiny
   discrete policy.
9. `06_frontier/05_reasoning_rl` — separate verifiable reward, exploration and
   inference-time sampling in a small arithmetic environment.

## Paper ladder

Use papers as specifications for experiments, not as reading trophies. The
following anchors give each later lesson a concrete source:

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
  and [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
  — the Skip-gram architecture, then negative sampling, frequent-word subsampling,
  the 3/4-power noise distribution, and linear embedding structure.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) —
  degradation in deep plain networks, residual reparameterisation, and economical
  identity shortcuts.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — scaled
  dot-product attention, multi-head attention and the encoder/decoder design.
- [BERT](https://aclanthology.org/N19-1423/) — deep bidirectional encoder
  pre-training, masked language modelling, sentence-pair inputs and minimal
  task-specific fine-tuning heads.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  — freeze a pretrained matrix, learn `ΔW=BA`, and merge the update for deployment.
- [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) —
  a falsifiable mechanistic account that can be tested in a small model.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  — turn scaling claims into controlled model/data sweeps.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — sparse expert routing
  with only a subset active for each token.
- [Mamba](https://arxiv.org/abs/2312.00752) — input-dependent state-space
  dynamics as an alternative sequence mechanism.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — alignment
  as a simple preference classification objective.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reasoning behaviour from
  reinforcement learning with verifiable rewards and a multi-stage recipe.

This list is deliberately small. Before adding a newer paper, identify which
mechanism it replaces or extends and what result can be reproduced on a laptop.

## Definition of a paper reproduction

A lesson is not a reproduction merely because the model runs. Record:

1. the paper's central causal or empirical claim in one sentence;
2. the smallest dataset/model that can still exhibit the claimed phenomenon;
3. one trusted baseline and the exact metric;
4. at least one ablation that should remove the effect;
5. seeds, device, runtime and dependency versions;
6. the gap between the small experiment and the paper's actual evidence.

If the result depends on a pretrained checkpoint or paid API, keep that path
optional and ship a deterministic toy experiment that still teaches the
mechanism.
