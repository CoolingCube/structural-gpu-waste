# CoolingCube — Structural GPU Waste Analysis

**Prove the GPU waste in your LLM inference cluster. Not an estimate — a lower bound.**

Most teams running self-hosted LLMs size GPU capacity for peak demand and pay
for idle time between bursts. This waste is **structural** — no batching trick,
framework tuning, or quantization eliminates it. The only fix is pooling.

## The math

```
min_waste = 1 - (avg_demand / peak_demand)
```

This is a proven lower bound. It holds regardless of your serving framework,
model size, or infrastructure. It follows directly from the definition of burstiness.

---

## Results on real production data

Three independent datasets. Three peer-reviewed papers. Same finding.

### 1. Azure LLM Inference 2023 — Text Services
*Splitwise (ISCA 2024) + DynamoLLM (HPCA 2025) · CC-BY*

| Service | Burstiness | Avg utilization | Structural waste | Monthly cost |
|---|---|---|---|---|
| Conversational | 1.6x | 62% | 17% | $1,800 |
| **Code** | **2.8x** | **35%** | **53%** | **$18,000** |
| Combined (pooled) | — | — | **44%** | **$19,800** |

### 2. Azure LMM Inference 2025 — Multimodal (Oct 2024)
*ModServe (SoCC 2025) · CC-BY · 1,000,000 requests · 7 days*

| Metric | Value |
|---|---|
| Request mix | 50% text-only / 50% with images |
| Burstiness | **3.8x** |
| Avg utilization | **26.6%** |
| Structural waste | **64%** |
| Monthly waste cost | **$57,600** |
| Reduction with pooling | 50 GPUs → 18 GPUs **(−64%)** |

> Images create sharper demand peaks while quiet periods stay the same length.
> Multimodal clusters are structurally more wasteful than text-only.

### 3. BurstGPT — Azure OpenAI Production Traces
*KDD 2025 · 10.5M requests · 121 days*

| Model | Burstiness | Idle fraction | Structural waste |
|---|---|---|---|
| ChatGPT | 6.3x | 61.7% | ~50% |
| GPT-4 | 4.1x | 64.3% | ~50% |
| Combined (pooled) | — | — | **50%** |

---

## The pattern across all datasets

| Dataset | Year | Burstiness | Waste |
|---|---|---|---|
| Azure text (conversational) | 2023 | 1.6x | 17% |
| Azure text (code) | 2023 | 2.8x | 53% |
| BurstGPT (ChatGPT) | 2022-23 | 6.3x | ~50% |
| BurstGPT (GPT-4) | 2022-23 | 4.1x | ~50% |
| **Azure multimodal** | **2024** | **3.8x** | **64%** |

Higher burstiness = more waste. Images make it worse.
This is the [Aegaeon](https://dl.acm.org/doi/10.1145/3694715.3695967) problem
(SOSP 2025) — Alibaba reduced 1,192 → 213 GPUs (−82%) with pooling.

---

## Run the analysis yourself

No login required for scripts 1 and 3. Script 2 requires HuggingFace login.

```bash
pip install pandas numpy matplotlib datasets
```

### Script 1 — Azure LLM text traces (2023)
```bash
python azure_waste_analysis.py
```
Downloads directly from [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset).

### Script 2 — Azure LMM multimodal traces (2025, most recent)
```bash
python azure_multimodal_waste.py
```
Downloads directly from [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset).
Most recent public Azure inference data. 1M requests, Oct 2024.

### Script 3 — BurstGPT (10M ChatGPT/GPT-4 traces)
```bash
python burstgpt_waste_analysis.py
```
Requires HuggingFace account + accepting [BurstGPT license](https://huggingface.co/datasets/lzzmm/BurstGPT).

---

## Get your cluster's number

The scripts above run on public Microsoft/HuggingFace data.

To measure **your specific cluster's** structural waste — from your vLLM,
SGLang, or TGI logs — visit [coolingcube.cc](https://coolingcube.cc).

You need: request timestamps, token counts, model/service name.
You get: your exact waste percentage, dollar cost, and a proof it's structural.

---

## Why this matters

- **Aegaeon** (Alibaba, SOSP 2025): 1,192 → 213 GPUs with pooling (−82%)
- **ModServe** (Microsoft, SoCC 2025): image/text disaggregation for multimodal
- **DynamoLLM** (Microsoft, HPCA 2025): energy-aware LLM cluster design
- Every team running dedicated GPU instances has this problem
- Most don't know their exact number — that's the gap

---

## Data sources & citations

```bibtex
@inproceedings{patel2024splitwise,
  title={Splitwise: Efficient generative LLM inference using phase splitting},
  author={Patel, Pratyush and Choukse, Esha and Zhang, Chaojie and ...},
  booktitle={ISCA 2024}
}

@inproceedings{stojkovic2025dynamollm,
  title={DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency},
  author={Stojkovic, Jovan and Zhang, Chaojie and Goiri, Inigo and ...},
  booktitle={HPCA 2025}
}

@inproceedings{qiu2025modserve,
  title={ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving},
  booktitle={SoCC 2025}
}

@inproceedings{wang2025burstgpt,
  title={BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems},
  booktitle={KDD 2025}
}
```

*All data used under CC-BY Attribution License.*
*Code: MIT License.*
