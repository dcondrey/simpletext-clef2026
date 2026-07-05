# Writerslogic at the CLEF 2026 SimpleText Track

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CLEF 2026 SimpleText](https://img.shields.io/badge/CLEF%202026-SimpleText-orange.svg)](http://simpletext-project.com/2026/)

Team `writerslogic` submission to the **CLEF 2026 SimpleText** track — Task 1 (scientific text simplification) and Task 2 (identify and avoid hallucination). Evaluated on English and multilingual biomedical text from Cochrane systematic reviews.

## Official Results

| Task | Metric | Score | Standing |
|---|---|---|---|
| **1.1** — sentence simplification | SARI / BLEU | **47.43** / 14.21 | best **sentence-level** system (3rd on the combined Task 1 board, behind two document-level systems) |
| **2.1** — binary overgeneration ID | doc-level macro F1 | **0.8085** | **2nd** of teams (behind AIIR Lab 0.8197); our DeBERTa identification run (0.8081) tops the identification track |
| **2.2** — multi-class error classification | accuracy | **0.804** | **2nd** among unique teams (behind AIIR Lab 0.827) |

## Task 1 — Multi-Candidate Simplification (`task1/`)

For each input sentence we generate **five candidates at varying temperatures** and select the best with a **reference-free SARI-proxy** score that rewards compression, source-word retention, Cochrane Plain-Language-Summary vocabulary, and lexical simplicity. Our best submission uses **Claude Sonnet 4** as the generator (GPT-4o-mini was used during development); a local Qwen3-8B variant trades quality for fully-local inference. The exact prompt and few-shot examples are in the paper.

## Task 2 — Overgeneration / Hallucination Detection (`task2/`)

We fine-tune **DeBERTa-v3-large (NLI)** on 350K `(source, sentence)` pairs, framing overgeneration detection as natural language inference (source = premise, candidate = hypothesis). This is complemented by an interpretable **LightGBM stacking ensemble** over 70+ features — source-alignment, prompt-leakage patterns, text-quality, repetition, and **sequential "CRF-like" features** that capture how overgeneration errors propagate once they begin within a document — with per-class threshold tuning for the imbalanced multi-class task.

## Repository Structure

```
task1/   # multi-candidate generation + SARI-proxy reranking
task2/   # DeBERTa-v3 NLI classifier + LightGBM stacking ensemble
```

## Citation

```bibtex
@inproceedings{condrey2026simpletext,
  title     = {Writerslogic at the {CLEF} 2026 {SimpleText} Track:
               Multi-Candidate {LLM} Simplification and Stacked Complexity Spotting},
  author    = {Condrey, David},
  booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the Evaluation Forum},
  series    = {CEUR Workshop Proceedings},
  year      = {2026},
  publisher = {CEUR-WS.org},
  note      = {to appear}
}
```
