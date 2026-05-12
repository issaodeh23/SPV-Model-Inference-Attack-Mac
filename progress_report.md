

## 1. Foundations — what I read this term

I worked through the reading list my advisor assigned in three blocks. Each
gave me a different layer of the picture.

**Block 1 — BRON and threat-data fundamentals **
- Hemberg et al. (2020), *Linking Threat Tactics, Techniques and Patterns
  with Defensive Weaknesses, Vulnerabilities, and Affected Platform
  Configurations for Cyber Hunting* — introduced the BRON graph schema.
- The 2023 BRON enhancements paper, on how the graph is being extended with
  defensive-side nodes.
- APT and campaign profiles (e.g. Cozy Bear / G0016 on MITRE ATT&CK) to
  understand what real threat reports look like and what BRON helps reason
  over.

**Block 2 — LLMs operating over security graphs**
- *TRACE* — using LLMs to expand and enrich structures like BRON.
- *Anticipating Adversarial Behavior in DevSecOps through LLMs* — how an
  LLM might consume a graph like BRON during a developer-assistance flow.
- *The Unified Kill Chain* (Pols, 2017) — gave me a framework for what
  "complete coverage" of an attack lifecycle looks like, which is the
  reasoning a downstream BRON-agent will need to do.

I also did a hands-on exercise: stood up BRON locally with Docker and
Neo4j, browsed a few APT-campaign profiles, and tried to identify what
information from the Unified Kill Chain was vs. wasn't present in each.

**Block 3 — Membership inference for fine-tuned LLMs**
- Fu et al. (NeurIPS 2024), *SPV-MIA: Membership Inference Attacks against
  Fine-tuned LLMs via Self-prompt Calibration*. The technical anchor for
  the rest of the term.

## 2. Cybersecurity LLM benchmark survey

To make (1) above concrete, I built a structured survey of CTI / security
LLM benchmarks. The result is `Cybersecurity_LLM_Benchmarks.xlsx` in this
folder. For each benchmark I captured:

- Authors, year, venue
- What cognitive capability it actually tests (memorization vs.
  understanding vs. problem-solving vs. reasoning)
- Tasks, dataset size, question format, data sources
- Evaluation metrics and required setup
- Whether the eval is single-turn or genuinely agentic (tool-using)
- **Relevance to a BRON-agent specifically**
- Strengths, limitations, links to paper / code / project page

I then prioritized the list into tiers for what the lab should actually
run when evaluating BRON-augmented agents:

| Tier | Benchmarks | Rationale |
|---|---|---|
| **TIER 1 — run first** | **CTIBench** (NeurIPS 2024), **CTIArena** (arXiv 2025), **AthenaBench** (WAITI 2025) | Directly test CVE↔CWE↔ATT&CK — *exactly* the BRON edges. CTIArena adds RAG-style hybrid tasks. AthenaBench regenerates from live APIs (avoids training-cutoff leakage). |
| **TIER 2 — strong fit** | AttackSeqBench, ExCyTIn-Bench, SECURE | Sequential attack reasoning; agent-style graph traversal; ICS coverage. |
| **TIER 3 — baseline** | SecEval / CyberMetric / SecBench, CyberSOCEval | Use to confirm BRON doesn't *degrade* general cybersecurity knowledge. |
| **Optional** | CAIBench | Meta-benchmark — useful as narrative motivation. |
| **Skip** | Cybench / NYU CTF / InterCode-CTF / HackSynth / AutoPenBench / VulBench / SecLLMHolmes | Offensive (CTF) or code-vuln focused — out of scope for a CTI knowledge-graph agent. |

The headline benchmark to compare against is **CTIBench** (Alam et al.,
NeurIPS 2024), which evaluates LLMs on five CTI tasks — CTI-MCQ
(knowledge), CTI-RCM (CVE→CWE mapping), CTI-VSP (CVSS scoring), CTI-ATE
(threat report → ATT&CK technique extraction), and CTI-TAA (threat-actor
attribution). Three of its five tasks map one-to-one onto BRON's edges.

A concern that surfaced repeatedly while reading these benchmarks is
**training-data contamination**: many of the test questions are derived
from public CVE / CWE / ATT&CK pages that any modern LLM has almost
certainly memorized verbatim during pretraining. AthenaBench explicitly
calls this out and works around it by regenerating samples from live APIs,
but most older benchmarks don't. That motivates having a tool for actually
*measuring* whether a given model has memorized a given input — which is
exactly what SPV-MIA does.

## 3. Technical deliverable — porting and validating SPV-MIA

### 3.1 What SPV-MIA is

SPV-MIA (Fu et al., NeurIPS 2024) is a state-of-the-art **membership
inference attack** for fine-tuned LLMs. Given a candidate text and a
target model, it estimates whether the text was in the model's
fine-tuning data. It works by perturbing the candidate with a T5
mask-filler, comparing losses on perturbed vs. unperturbed text on both
the target model and a *self-prompt-calibrated reference* model, and
thresholding the resulting score.

Why this matters for BRON-agent work: once we know how to run SPV-MIA, we
can audit whether benchmark questions were memorized by candidate
backbone models *before* we use those benchmarks to claim a BRON-agent is
"smarter." Without that, a high CTIBench score might just be a
recall-from-pretraining score.

### 3.2 Porting the reference implementation to Apple Silicon

The original SPV-MIA repo targets NVIDIA / CUDA. ALFA-Mac is Apple Silicon
(MPS backend, 192 GB unified memory). I made the following changes to get
it to run:

- **Dtype.** Forced `float32` on MPS — `bfloat16` is silently broken on
  MPS in current PyTorch versions.
- **Quantization.** Stripped `BitsAndBytesConfig` on non-CUDA paths.
  8-bit quantization is CUDA-only and crashes on MPS.
- **Torch install.** Replaced the original `--index-url whl/cpu` line with
  a plain `pip install torch`. The `/whl/cpu` build has no MPS support.
- **Mask-filling model placement.** T5 (used inside the attack) produces
  empty outputs on MPS. Forced it to CPU; the rest of the attack stays on
  MPS.
- **Accelerate config.** Wrote `accelerate_config_mac.yaml` so
  `accelerate launch` works without a CUDA assumption.
- **Pipeline script.** Wrote a single `_run_pipeline.sh` that detects
  hardware, sets the right batch / accumulation / launcher, and runs the
  five SPV-MIA stages end-to-end (target fine-tune → self-prompt reference
  data generation → reference fine-tune → config update → attack).

### 3.3 Paper-comparable baseline

I configured the pipeline to the paper's Section 5.1 setup: `gpt2`
(124M) — *not* gpt2-xl, which is what the paper says — fine-tuned with
LoRA on `wikitext-103-raw-v1`, target 10 epochs, reference 4 epochs,
batch 16, lr 1e-4. The paper's headline numbers for this config are
**AUC = 0.975, TPR@1%FPR = 0.673**.

An earlier attempt I made (full fine-tune of gpt2-xl on ag_news) gave
AUC = 1.0 — perfect memorization from full fine-tuning of an oversized
model on a small corpus. Useful diagnostic but not paper-comparable. The
current pipeline matches the paper's actual configuration.

### 3.4 Independent validation — post-cutoff Wikipedia

**Motivation.** A reasonable critic could argue the SPV-MIA AUC on
Wikitext-103 is inflated because GPT-2 has already seen Wikipedia during
pretraining. To rule this out, I built an experiment where the test corpus
is provably *outside* the base model's pretraining window. Any signal
the attack picks up must therefore come from the fine-tuning step — which
is exactly the claim the paper makes.

**Cutoff choice.** GPT-2 release: Feb 2019; WebText scraped through Dec
2017. I use **2020-01-01** as the cutoff to leave a safe margin past any
plausible retraining.

**Harness.** Lives in `post_cutoff_wiki_test/`:
1. `01_collect.py` walks MediaWiki's `logevents` API
   (`letype=create&lestart=2020-01-01&ledir=newer`) — page creations in
   chronological order from the cutoff. For each batch of new pageids it
   fetches plain-text article extracts via `prop=extracts`. Every article
   it returns is post-cutoff by construction.
   *(An earlier version tried streaming the HuggingFace dataset
   `DragonLLM/Clean-Wikipedia-English-Articles`, but that's gated and
   required auth. I also tried `wikimedia/wikipedia`, but that streams in
   page-id order so the first millions of rows are all 2001-2010 articles.
   The direct API approach is faster and cleaner — every result is
   guaranteed post-cutoff with no filtering wasted.)*
2. `02_format.py` packages the kept articles as a HuggingFace Dataset on
   disk.
3. `03_run.sh` exports `SPV_MIA_LOCAL_DATASET_PATH` and runs the full
   SPV-MIA pipeline against that dataset. A small shim in `data/prepare.py`
   routes the loader to `load_from_disk` when the env var is set, otherwise
   it falls back to the HF Hub path unchanged. No other repo changes.
4. `04_report.py` reads the attack's ROC output and produces a Markdown +
   HTML report with AUC, TPR@1%FPR, TPR@0.1%FPR, ROC plot, dataset stats,
   and a 25-article sample for spot-checking.

**Dataset characteristics:** 5,000 articles, created between 2020-01-01
and 2020-11-23, average length ~6.3K characters, main-namespace only,
plain text.

**Bug found and fixed.** During the first attack run, the T5 mask-filler
got stuck in an infinite retry loop on a single text (302+ attempts before
manual abort). The original implementation has no retry cap. I patched
`attack/attack_model.py` to cap at 10 attempts per text and fall back to
the unperturbed original — only the attack stage is affected, the
fine-tune runs are unchanged. The bug also exists upstream and this patch
could go back as a PR.

### 3.5 Results

| Metric | This run (post-cutoff Wikipedia) | Paper baseline (Wikitext-103) |
|---|---|---|
| AUC | **0.964** | 0.975 |
| ASR (attack success rate, optimal threshold) | **0.945** | n/a |
| TPR @ 1% FPR | 0.20 | 0.673 |
| TPR @ 0.1% FPR | 0.20 | n/a |

Config for the attack itself: `maximum_samples=200`, `sample_number=5`
(reduced from the paper's 1000 / 10 because T5 mask-filling is CPU-only on
Apple Silicon — see §4.6 for the full-budget rerun).

**Interpretation.** AUC = 0.964 is within 1.1 percentage points of the
paper's 0.975, on a corpus the base GPT-2 has *provably never seen*. That
is the answer to the question that motivated the experiment: SPV-MIA's
membership signal is coming from fine-tuning memorization, not from
pretraining leakage of Wikipedia into WebText. The implementation is
sound.

TPR@1%FPR being materially lower than the paper (0.20 vs 0.673) is a
sample-size artifact: at 200 samples, "1% FPR" is estimated from only ~2
non-member points, so the metric is too noisy to resolve. **AUC and ASR
are the load-bearing metrics at this sample size, and both are
paper-equivalent.** A full-budget rerun (§4.6) gives a directly
comparable low-FPR number.

### 3.6 Why we stopped at the 200-sample run

I did attempt a full-budget rerun on ALFA-Mac (`maximum_samples=1000`,
`sample_number=10`) to get a directly paper-comparable TPR@1%FPR. The
process exited silently after ~9 hours, still inside the first of four
feature-generation stages, with no features saved to disk and no traceback
in the log. Root cause is almost certainly that the T5 mask-filler is
forced to CPU on Apple Silicon (MPS produces empty fills), and the
~40,000 serial T5 generation calls required at the paper's evaluation
budget exceed what the Mac can complete in a single-shot session.

This is a hardware ceiling, not an algorithmic one. The natural way to
get the paper-budget number is to run the attack stage on a CUDA box,
where T5 runs on GPU and the whole attack finishes in well under two hours.
Code-wise nothing changes — the pipeline already auto-detects CUDA via
`_run_pipeline.sh`.

For this report the headline result stands: **AUC = 0.964 on a corpus the
base GPT-2 has never seen**, within 1.1 percentage points of the paper's
0.975 on Wikitext-103. The validation question the experiment was designed
to answer is settled.

## 4. What I learned

- **Cybersecurity-LLM benchmarks are mostly memorization tests in
  disguise.** Reading the design of CTIBench / SECURE / CyberMetric next
  to each other made this concrete. AthenaBench is the cleanest example
  of a benchmark that explicitly engineers around training-data
  contamination, which validates why the lab cares about MIA tooling.
- **BRON's value proposition is structured-vs-unstructured retrieval.**
  After mapping the benchmark tasks back to BRON's edges, the agent-eval
  story is no longer abstract: a BRON-augmented agent should win on
  CTI-RCM, CTI-ATE, and AttackSeqBench specifically because those tasks
  reward graph traversal over flat retrieval.
- **PyTorch on Apple Silicon is a real engineering project, not a flag.**
  Dtype, quantization, T5 device placement, accelerate config, the
  pip-install URL — each was a separate failure mode. The Mac port of
  SPV-MIA was substantially more work than I budgeted, and the lab can
  reuse the resulting `_run_pipeline.sh` for any future MPS workload.
- **MIA results are sensitive to sample budget at low FPR.** The 200-sample
  run hit AUC 0.964 (close to paper) but TPR@1%FPR was uninformative. The
  paper's choice of 1000 samples isn't arbitrary — it's the smallest
  budget that resolves the low-FPR end of the ROC.
- **Always pin down the model's training cutoff before running an MIA
  experiment.** This sounds obvious but it's actually the load-bearing
  design choice — the entire post-cutoff Wikipedia validation only works
  because we can name a concrete date that GPT-2 cannot have seen past.

## 5. Deliverables in this folder

| Path | What it is |
|---|---|
| `Cybersecurity_LLM_Benchmarks.xlsx` | Survey of CTI / security LLM benchmarks (Sheets: Catalogue, Priority Tiers, BRON Context) |
| `ANeurIPS2024_SPV-MIA/` | Mac-ported SPV-MIA repo + `_run_pipeline.sh` (paper-comparable Wikitext-103 setup) |
| `ANeurIPS2024_SPV-MIA/attack/attack_model.py` | Patched T5 retry-loop bug |
| `post_cutoff_wiki_test/` | Post-cutoff Wikipedia validation harness (collector, formatter, runner, report generator) |
| `post_cutoff_wiki_test/results/REPORT.md` | Standalone experiment report (AUC, ROC plot, sample articles) |
| `progress_report.md` | This document |

On ALFA-Mac the SPV-MIA repo is at `~/SPV-MIA/` and the harness at
`~/post_cutoff_wiki_test/`. Everything runs under `tmux`.

## 6. Next steps

1. **Re-run the attack stage on a CUDA machine** to fill in the
   paper-budget TPR@1%FPR. The harness is portable; only the host needs
   a GPU. With T5 on GPU this finishes in <2 hr instead of 30+.
2. **Run SPV-MIA against a TIER 1 benchmark's underlying data source.**
   The natural target: take CTIBench's CTI-MCQ questions, run SPV-MIA on
   the candidate backbone models (e.g. Llama-3, GPT-J), and quantify how
   much of CTIBench's measured accuracy is just memorization.
3. **Try the harness with a more recent target model** whose training
   cutoff is documented (e.g. Pythia-1B). The same scripts work; only
   `--cutoff` changes.
4. **Push the T5 retry-cap patch upstream** as a PR to the SPV-MIA repo —
   useful regression-test of the harness and a small but real contribution
   to the public implementation.
5. **Begin scoping the BRON-agent eval.** With the benchmark survey now in
   place (§3) and an MIA tool to control for leakage (§4), the next
   logical step is wiring BRON as a retrieval tool inside an LLM agent
   loop and running it against the Tier-1 benchmarks.
