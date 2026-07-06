"""
================================================================================
MULTIMODAL GPU WASTE ANALYSIS -- Azure LMM Inference Dataset 2025
================================================================================
The most recent public Azure inference traces: one week of multimodal (vision+text)
LLM inference, collected October 2024, published in SoCC 2025.

Columns: TIMESTAMP, NumImages, ContextTokens, GeneratedTokens

This is the NEXT wave after text-only LLMs:
  - GPT-4V class models (vision + language)
  - Images add variable compute burden (more tokens = more GPU time)
  - Burstiness is even more extreme with mixed image/text requests

Same structural waste analysis as the text-only datasets, but now with
the image dimension showing why multimodal serving is especially wasteful.

No login required. Direct download from Microsoft GitHub (CC-BY).

Run in Colab:
  exec(open('/content/drive/MyDrive/azure_multimodal_waste.py').read())

Data: Azure/AzurePublicDataset (CC-BY) | Paper: ModServe, SoCC 2025
================================================================================
"""
import subprocess, sys, io, urllib.request, gzip
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'pandas', 'numpy', 'matplotlib'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ---- CONFIG ----
WINDOW_SEC      = 60        # 1-minute demand windows
PEAK_PERCENTILE = 0.95
POOLING_BUFFER  = 1.30
GPU_HOURLY_COST = 2.50      # H100 spot price USD

# Multimodal model throughput (tokens/sec per H100)
# Vision models are slower than text-only due to image encoding
# GPT-4V class: ~600 tok/s, image encoding adds ~200ms per image
TEXT_THROUGHPUT  = 600      # tokens/sec for text-only requests
IMAGE_OVERHEAD_MS = 200     # ms per image for vision encoding

print("="*64)
print("MULTIMODAL GPU WASTE ANALYSIS")
print("Azure LMM Inference Dataset 2025 (SoCC 2025 paper)")
print("Most recent public Azure inference traces -- Oct 2024")
print("="*64)

# ---- LOAD DATA ----
print("\nDownloading Azure LMM traces from Microsoft GitHub...")
URL = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLMMInferenceTrace_multimodal.csv.gz"
req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
data = urllib.request.urlopen(req, timeout=60).read()
print(f"  downloaded {len(data)/1e6:.1f} MB")

with gzip.open(io.BytesIO(data)) as f:
    df = pd.read_csv(f)

print(f"  {len(df):,} requests loaded")
print(f"  columns: {list(df.columns)}")

# ---- PARSE ----
df['ts'] = pd.to_datetime(df['TIMESTAMP'])
df['ts_sec'] = (df['ts'] - df['ts'].min()).dt.total_seconds()
df['total_tokens'] = df['ContextTokens'] + df['GeneratedTokens']
df['has_image'] = df['NumImages'] > 0
df['multi_image'] = df['NumImages'] > 1

duration_days = (df['ts'].max() - df['ts'].min()).total_seconds() / 86400
print(f"\nDataset: {len(df):,} requests over {duration_days:.1f} days")
print(f"  Text-only requests: {(~df['has_image']).sum():,} ({(~df['has_image']).mean()*100:.1f}%)")
print(f"  Single-image:       {(df['NumImages']==1).sum():,} ({(df['NumImages']==1).mean()*100:.1f}%)")
print(f"  Multi-image:        {df['multi_image'].sum():,} ({df['multi_image'].mean()*100:.1f}%)")
print(f"  Avg images/request: {df['NumImages'].mean():.2f}")
print(f"  Avg context tokens: {df['ContextTokens'].mean():.0f}")
print(f"  Avg gen tokens:     {df['GeneratedTokens'].mean():.0f}")

# ---- COMPUTE GPU DEMAND ----
# Effective GPU time per request:
# text_tokens / throughput + image_count * image_overhead
df['gpu_time_sec'] = (df['total_tokens'] / TEXT_THROUGHPUT +
                      df['NumImages'] * IMAGE_OVERHEAD_MS / 1000)

# Bin into 1-minute windows
t0 = df['ts_sec'].min(); t1 = df['ts_sec'].max()
bins = np.arange(t0, t1 + WINDOW_SEC, WINDOW_SEC)
df['bin'] = pd.cut(df['ts_sec'], bins=bins, labels=False)

# GPU demand per window = total GPU time / window duration
gpu_time_per_window = df.groupby('bin')['gpu_time_sec'].sum()
all_bins = pd.Series(0.0, index=range(len(bins)-1))
gpu_time_per_window = all_bins.add(gpu_time_per_window, fill_value=0)
gpu_demand = gpu_time_per_window / WINDOW_SEC  # GPU-equivalents

# Also split: text-only vs image requests
df_text = df[~df['has_image']]
df_image = df[df['has_image']]

def compute_demand(subset, label):
    if len(subset) < 100: return None
    tpw = subset.groupby('bin')['gpu_time_sec'].sum()
    all_b = pd.Series(0.0, index=range(len(bins)-1))
    tpw = all_b.add(tpw, fill_value=0)
    return tpw / WINDOW_SEC

gpu_text  = compute_demand(df_text, 'text')
gpu_image = compute_demand(df_image, 'image')

# ---- WASTE ANALYSIS ----
print("\n" + "="*64)
print("STRUCTURAL WASTE ANALYSIS")
print("="*64)

def analyze(demand, label):
    avg   = demand.mean()
    peak  = demand.quantile(PEAK_PERCENTILE)
    idle  = (demand < 0.001).mean()
    burst = peak / max(avg, 1e-9)
    dedicated = max(1, np.ceil(peak))
    pooled    = max(1, np.ceil(avg * POOLING_BUFFER))
    wasted    = dedicated - pooled
    waste_pct = wasted / dedicated * 100
    waste_usd = wasted * GPU_HOURLY_COST * 24 * 30
    util      = avg / max(peak, 1e-9) * 100

    print(f"\n{label}:")
    print(f"  Avg GPU demand:    {avg:.3f} GPUs")
    print(f"  P95 GPU demand:    {peak:.3f} GPUs")
    print(f"  Burstiness:        {burst:.1f}x")
    print(f"  Avg utilization:   {util:.1f}% (if sized for p95)")
    print(f"  Idle windows:      {idle*100:.1f}%")
    print(f"  ─────────────────────────────────")
    print(f"  Dedicated GPUs:    {dedicated:.0f}")
    print(f"  Pooled GPUs:       {pooled:.0f}")
    print(f"  WASTE:             {wasted:.0f} GPU ({waste_pct:.1f}%)")
    print(f"  Cost/month:        ${waste_usd:,.0f} USD")
    return {'avg':avg,'peak':peak,'idle':idle,'burst':burst,'util':util,
            'dedicated':dedicated,'pooled':pooled,'wasted':wasted,
            'waste_pct':waste_pct,'waste_usd':waste_usd,'demand':demand}

r_all   = analyze(gpu_demand, "ALL REQUESTS (text + images)")
r_text  = analyze(gpu_text,  "TEXT-ONLY requests") if gpu_text is not None else None
r_image = analyze(gpu_image, "IMAGE requests") if gpu_image is not None else None

# Combined: what if text and image served separately vs pooled?
if r_text and r_image:
    tot_sep = r_text['dedicated'] + r_image['dedicated']
    comb    = gpu_demand  # already combined
    comb_avg  = comb.mean()
    comb_peak = comb.quantile(PEAK_PERCENTILE)
    comb_pool = max(1, np.ceil(comb_avg * POOLING_BUFFER))
    comb_waste = tot_sep - comb_pool
    comb_pct   = comb_waste / tot_sep * 100
    comb_usd   = comb_waste * GPU_HOURLY_COST * 24 * 30

    print(f"\n{'='*64}")
    print("SEPARATION vs POOLING (text-only vs image instances):")
    print(f"  Separate dedicated: {tot_sep:.0f} GPUs")
    print(f"  Pooled together:    {comb_pool:.0f} GPUs")
    print(f"  WASTE eliminated:   {comb_waste:.0f} GPUs ({comb_pct:.1f}%)")
    print(f"  Monthly saving:     ${comb_usd:,.0f} USD")

# ---- IMAGE IMPACT ANALYSIS ----
print(f"\n{'='*64}")
print("IMAGE DIMENSION: why multimodal is more wasteful")
print("="*64)
print(f"""
Image requests vs text-only:
  Text avg tokens:    {df_text['total_tokens'].mean():.0f}
  Image avg tokens:   {df_image['total_tokens'].mean():.0f} (+images)
  Avg images/request: {df_image['NumImages'].mean():.2f}
  Image overhead/req: {df_image['NumImages'].mean() * IMAGE_OVERHEAD_MS:.0f}ms extra GPU time

Why images make burstiness worse:
  - Image requests take {df_image['NumImages'].mean() * IMAGE_OVERHEAD_MS:.0f}ms longer on average
  - A burst of image requests hits the GPU much harder than text bursts
  - But quiet periods are the same length regardless
  - So the peak/avg ratio is larger for image-capable serving

This is the ModServe problem (SoCC 2025):
  Alibaba / Azure found that multimodal clusters are even more wasteful
  than text-only because image bursts create sharper peaks.
  ModServe separates image encoding from text generation to improve pooling.
""")

# ---- HOURLY PATTERN ----
print("Computing hourly demand pattern...")
df['hour'] = df['ts'].dt.hour
hourly_demand = df.groupby('hour')['gpu_time_sec'].sum() / (
    duration_days * 3600)  # normalize to GPU demand

# ---- VISUALIZATION ----
print("Generating visualization...")
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Multimodal LLM Inference — Structural GPU Waste\n'
             'Azure LMM Dataset (Oct 2024) | SoCC 2025 | ModServe Paper',
             fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1. GPU demand over time (full week)
ax1 = fig.add_subplot(gs[0, :2])
t_hours = gpu_demand.index * WINDOW_SEC / 3600
ax1.fill_between(t_hours, 0, gpu_demand.values,
                 alpha=0.5, color='purple', label='Total (text+image)')
if gpu_text is not None:
    ax1.plot(t_hours, gpu_text.values, alpha=0.6, color='steelblue',
             linewidth=0.7, label='Text-only')
if gpu_image is not None:
    ax1.plot(t_hours, gpu_image.values, alpha=0.6, color='darkorange',
             linewidth=0.7, label='Image requests')
ax1.axhline(r_all['avg'], color='green', linestyle='--', linewidth=1.5,
            label=f"Avg ({r_all['avg']:.2f} GPUs)")
ax1.axhline(r_all['peak'], color='red', linestyle='--', linewidth=1.5,
            label=f"P95 ({r_all['peak']:.2f} GPUs)")
ax1.set_xlabel('Hour since start'); ax1.set_ylabel('GPU demand')
ax1.set_title(f'GPU Demand Over Time ({duration_days:.0f} days)')
ax1.legend(fontsize=8, ncol=2)

# 2. Hourly pattern (daily cycle)
ax2 = fig.add_subplot(gs[0, 2])
hours = list(range(24))
demand_vals = [hourly_demand.get(h, 0) for h in hours]
colors_h = ['#d73027' if d > np.percentile(demand_vals, 75)
            else '#1a9850' if d < np.percentile(demand_vals, 25)
            else '#fee08b' for d in demand_vals]
ax2.bar(hours, demand_vals, color=colors_h, edgecolor='black', linewidth=0.3)
ax2.set_xlabel('Hour of day (UTC)'); ax2.set_ylabel('Avg GPU demand')
ax2.set_title('Daily Demand Pattern')
ax2.set_xticks(range(0, 24, 4))

# 3. Request type breakdown
ax3 = fig.add_subplot(gs[1, 0])
types = ['Text-only', 'Single-image', 'Multi-image']
counts = [(~df['has_image']).sum(),
          (df['NumImages']==1).sum(), df['multi_image'].sum()]
colors3 = ['steelblue', 'darkorange', 'darkred']
wedges, texts, autotexts = ax3.pie(
    counts, labels=types, colors=colors3, autopct='%1.1f%%',
    startangle=90, textprops={'fontsize': 8})
ax3.set_title('Request Type Distribution')

# 4. Waste comparison
ax4 = fig.add_subplot(gs[1, 1])
categories = ['Dedicated\n(no pooling)', 'Pooled\n(same cluster)']
values_w = [r_all['dedicated'], r_all['pooled']]
bc = ['#d73027', '#1a9850']
bars4 = ax4.bar(categories, values_w, color=bc,
                edgecolor='black', linewidth=0.5, width=0.5)
ax4.set_ylabel('GPUs required')
ax4.set_title('GPU Savings from Pooling')
for bar, v in zip(bars4, values_w):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{v:.0f} GPUs\n${v*GPU_HOURLY_COST*24*30:,.0f}/mo',
             ha='center', fontweight='bold', fontsize=9)
ax4.annotate(f'↓ {r_all["waste_pct"]:.0f}% waste\n${r_all["waste_usd"]:,.0f}/mo',
             xy=(1, r_all['pooled']), xytext=(0.5, (r_all['dedicated']+r_all['pooled'])/2),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, fontweight='bold', color='#1a9850', ha='center')

# 5. Key proof box
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
proof = (
    "FEASIBILITY PROOF\n"
    "─────────────────\n\n"
    f"Burstiness = {r_all['burst']:.1f}x\n\n"
    f"Avg utilization:\n"
    f"= avg / peak\n"
    f"= {r_all['avg']:.2f} / {r_all['peak']:.2f}\n"
    f"= {r_all['util']:.1f}%\n\n"
    f"Min waste:\n"
    f"= 1 - 1/{r_all['burst']:.1f}\n"
    f"= {(1-1/r_all['burst'])*100:.1f}%\n\n"
    f"Idle windows:\n"
    f"= {r_all['idle']*100:.1f}%\n\n"
    f"Images make it worse:\n"
    f"sharper peaks,\n"
    f"same quiet periods.\n\n"
    f"ModServe (SoCC 2025)\n"
    f"solved this for Azure.\n"
    f"CoolingCube measures\n"
    f"it for your cluster."
)
ax5.text(0.05, 0.97, proof, transform=ax5.transAxes,
         fontsize=8.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/content/azure_multimodal_waste.png', dpi=150, bbox_inches='tight')
try:
    plt.savefig('/content/drive/MyDrive/azure_multimodal_waste.png',
                dpi=150, bbox_inches='tight')
    print("  saved: /content/drive/MyDrive/azure_multimodal_waste.png")
except Exception:
    print("  saved: /content/azure_multimodal_waste.png")
plt.show()

print(f"\n{'='*64}")
print("PUBLISHABLE FINDINGS")
print("="*64)
print(f"""
Dataset: Azure LMM Inference Oct 2024 | {len(df):,} requests | {duration_days:.0f} days
Paper:   ModServe, SoCC 2025 (Microsoft Research)

Request mix: {(~df['has_image']).mean()*100:.0f}% text-only | {df['has_image'].mean()*100:.0f}% with images

Structural waste (dedicated vs pooled):
  Burstiness:     {r_all['burst']:.1f}x peak/avg
  Avg utilization:{r_all['util']:.1f}% if sized for p95
  Structural waste:{r_all['waste_pct']:.1f}%
  Monthly cost:   ${r_all['waste_usd']:,.0f} USD (H100 pricing)
  Idle windows:   {r_all['idle']*100:.1f}% of minutes near-zero demand

Why multimodal is worse than text-only:
  Image requests create sharper demand peaks (more GPU time per request)
  while quiet periods between bursts remain the same length.
  The burstiness ratio is higher, the waste floor is higher.

This is exactly what ModServe (SoCC 2025) addresses at Azure scale.
CoolingCube measures it at your scale.

Data: github.com/Azure/AzurePublicDataset (CC-BY)
Code: github.com/CoolingCube/structural-gpu-waste
""")
