# CoolingCube

**Structural GPU waste analysis for LLM inference clusters.**

Most teams running self-hosted LLMs size their GPU capacity for peak demand
and pay for idle time between bursts. This waste is structural — no batching
trick, framework tuning, or quantization eliminates it. You need to know your
number before you can fix it.

## What is structural GPU waste?

When demand is bursty (it always is for LLM workloads), a dedicated GPU
instance sits idle between bursts. The minimum waste is:

```
min_waste = 1 - (avg_demand / peak_demand)
```

This is a **proven lower bound**, not an estimate. It holds regardless of
your serving framework, model size, or infrastructure setup.

## Results on real Azure production traces

Running this analysis on Microsoft's public Azure LLM inference dataset
(28,185 requests, real production data, CC-BY license):

| Service | Burstiness | Avg utilization | Structural waste | Monthly cost |
|---|---|---|---|---|
| Conversational | 1.6x | 62% | 17% | $1,800 |
| Code | **2.8x** | **35%** | **53%** | **$18,000** |
| Combined (pooled) | — | — | **44%** | **$19,800** |

The code service runs at **34.6% average utilization** when sized for peak.
Pooling both services recovers 44% of the total GPU budget.

This is the same pattern Alibaba found at 1,192 GPUs and fixed with
[Aegaeon](https://dl.acm.org/doi/10.1145/3694715.3695967) (SOSP 2025),
reducing their cluster from 1,192 → 213 GPUs (-82%).

## Run the analysis on public data

```bash
pip install pandas numpy matplotlib
python azure_waste_analysis.py
```

Downloads Microsoft's public Azure LLM inference traces directly from
[Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)
(CC-BY Attribution License) and produces the waste analysis + chart.

## Get your cluster's number

The public analysis runs on Microsoft's data. To measure **your specific
cluster's** structural waste ratio — from your vLLM, SGLang, or TGI logs —
visit [coolingcube.cc](https://coolingcube.cc).

You get: your exact waste percentage, the dollar cost, and a proof that it's
structural (not fixable by tuning alone).

## Why this matters now

- Aegaeon proved the problem exists at hyperscale (Alibaba, SOSP 2025)
- DynamoLLM quantified it on Azure production traces (Microsoft, HPCA 2025)  
- BurstGPT characterized it across 10M ChatGPT/GPT-4 requests (KDD 2025)
- Every team running dedicated GPU instances has this problem
- Most don't know their exact number

---

*Data: [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) (CC-BY)*  
*Method: Aegaeon (SOSP 2025), DynamoLLM (HPCA 2025)*
