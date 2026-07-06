# Methodology

## The structural waste argument

LLM inference demand is bursty by nature. Requests arrive in clusters
separated by quiet periods. The burstiness ratio is:

```
burstiness = peak_demand / avg_demand
```

A dedicated GPU instance must be sized for peak demand to avoid dropping
requests. But between bursts, it sits idle. The minimum idle fraction is:

```
min_idle = 1 - (1 / burstiness)
```

This is a **lower bound** — the actual idle fraction is at least this large,
regardless of how well you optimize batching, quantization, or scheduling.
It follows directly from the definition of burstiness.

For the Azure code service (burstiness = 2.8x):

```
min_idle = 1 - (1 / 2.8) = 64.3%
```

In practice the measured idle fraction was 20.7% of 1-minute windows near
zero, and average utilization was 34.6%. The gap between 34.6% and 100% is
the structural waste.

## Why no single-instance trick fixes it

Within a single dedicated GPU instance, you can improve utilization through:
- Continuous batching (vLLM's default)
- KV cache optimization
- Quantization (more throughput per GPU-second)
- Speculative decoding

These reduce the GPU-seconds needed per request. They do **not** change the
burstiness of the arrival process. The periods of near-zero demand remain.
The structural waste floor is determined by the demand distribution, not the
serving efficiency.

## What does fix it

Pooling across services with non-correlated demand patterns. When the code
service is in a quiet period, those GPU-seconds can serve conversational
requests, and vice versa. The combined peak is lower than the sum of
individual peaks (anti-correlation benefit).

This is the Aegaeon approach: token-level scheduling across a shared GPU pool,
so no GPU-second is wasted on one service's quiet period when another has
active requests.

## How to measure your cluster

The public script (`azure_waste_analysis.py`) runs on Microsoft's open data.
To apply the same analysis to your cluster you need:

1. Request timestamps (when each request arrived)
2. Token counts (input + output per request)
3. Which model/service handled each request

Most serving frameworks log this by default:
- **vLLM**: request logs include timestamp, prompt_tokens, completion_tokens
- **SGLang**: similar request-level logging
- **TGI** (maintenance mode as of Dec 2025): access logs with token counts

With those three fields, the burstiness ratio and structural waste floor
are computable in the same way as the Azure analysis.

## Data sources

- Azure LLM Inference Dataset 2023: [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)
  (CC-BY, cited in Splitwise ISCA 2024 and DynamoLLM HPCA 2025)
- BurstGPT: [lzzmm/BurstGPT](https://huggingface.co/datasets/lzzmm/BurstGPT)
  (cited in KDD 2025)
- Aegaeon: [SOSP 2025](https://dl.acm.org/doi/10.1145/3694715.3695967)
  — Alibaba Cloud GPU pooling system
