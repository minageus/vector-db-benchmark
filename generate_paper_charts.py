import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import argparse

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

MILVUS_COLOR = '#2E86AB'
WEAVIATE_COLOR = '#E94F37'
COLORS = {'Milvus': MILVUS_COLOR, 'Weaviate': WEAVIATE_COLOR}
MARKERS = {'Milvus': 'o', 'Weaviate': 's'}

CURATED_FILES = {
    'fashion-mnist-784':  'paper_benchmark_20260201_210904.json',   # 60K, 784D, L2
    'gist1m':             'paper_benchmark_20260201_163717.json',   # 1M, 960D, L2
    'sift1m':             'paper_benchmark_20260201_184237.json',   # 1M, 128D, L2
    'glove-25':           'paper_benchmark_20260201_195809.json',   # 1.18M, 25D, cosine
    'glove-100':          'paper_benchmark_20260201_230242.json',   # 1.18M, 100D, cosine
    'glove-200':          'paper_benchmark_20260202_085252.json',   # 1.18M, 200D, cosine
    'nytimes-256':        'paper_benchmark_20260201_215039.json',   # 290K, 256D, cosine
    'deep-image-96-2M':   'paper_benchmark_20260205_143205.json',   # 2M, 96D, L2
    'deep-image-96-5M':   'paper_benchmark_20260205_184103.json',   # 5M, 96D, L2
}

DATASET_META = {
    'fashion-mnist-784':  {'vectors': 60000,   'dim': 784,  'metric': 'L2',     'label': 'F-MNIST\n60K x 784'},
    'gist1m':             {'vectors': 1000000, 'dim': 960,  'metric': 'L2',     'label': 'GIST\n1M x 960'},
    'sift1m':             {'vectors': 1000000, 'dim': 128,  'metric': 'L2',     'label': 'SIFT\n1M x 128'},
    'glove-25':           {'vectors': 1183514, 'dim': 25,   'metric': 'cosine', 'label': 'GloVe-25\n1.2M x 25'},
    'glove-100':          {'vectors': 1183514, 'dim': 100,  'metric': 'cosine', 'label': 'GloVe-100\n1.2M x 100'},
    'glove-200':          {'vectors': 1183514, 'dim': 200,  'metric': 'cosine', 'label': 'GloVe-200\n1.2M x 200'},
    'nytimes-256':        {'vectors': 290000,  'dim': 256,  'metric': 'cosine', 'label': 'NYTimes\n290K x 256'},
    'deep-image-96-2M':   {'vectors': 2000000, 'dim': 96,   'metric': 'L2',     'label': 'Deep-96\n2M x 96'},
    'deep-image-96-5M':   {'vectors': 5000000, 'dim': 96,   'metric': 'L2',     'label': 'Deep-96\n5M x 96'},
}

DISPLAY_ORDER = [
    'fashion-mnist-784', 'nytimes-256', 'glove-25', 'glove-100',
    'glove-200', 'sift1m', 'gist1m', 'deep-image-96-2M', 'deep-image-96-5M',
]


def _save(fig, output_dir: Path, name: str):
    for fmt in ['png', 'pdf']:
        fig.savefig(output_dir / f'{name}.{fmt}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  [OK] {name}.png / .pdf")


def _get_agg(result: Dict, key: str, field: str):
    """Extract a numeric field from the aggregated list for a given key (e.g. 'k10')."""
    for entry in result.get('aggregated', []):
        if entry.get('K') == key:
            return entry.get(field, 0)
    return 0


def load_curated(results_dir: Path) -> Dict[str, Dict]:
    """Load curated JSON files into {dataset_name: data} dict."""
    data = {}
    for ds_name, filename in CURATED_FILES.items():
        path = results_dir / filename
        if path.exists():
            with open(path) as f:
                data[ds_name] = json.load(f)
        else:
            print(f"  [WARN] Missing: {filename}")
    return data

def fig_loading(results: Dict[str, Dict], output_dir: Path):
    datasets = [d for d in DISPLAY_ORDER if d in results]
    labels = [DATASET_META[d]['label'] for d in datasets]

    milvus_time = [results[d]['loading']['Milvus']['load_time_seconds'] for d in datasets]
    weav_time   = [results[d]['loading']['Weaviate']['load_time_seconds'] for d in datasets]

    n_vecs = [DATASET_META[d]['vectors'] for d in datasets]
    milvus_tput = [n / t for n, t in zip(n_vecs, milvus_time)]
    weav_tput   = [n / t for n, t in zip(n_vecs, weav_time)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(datasets))
    w = 0.35

    ax1.bar(x - w/2, milvus_time, w, label='Milvus', color=MILVUS_COLOR)
    ax1.bar(x + w/2, weav_time,   w, label='Weaviate', color=WEAVIATE_COLOR)
    ax1.set_ylabel('Load Time (seconds)')
    ax1.set_title('(a) Data Loading Time')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')

    ax2.bar(x - w/2, milvus_tput, w, label='Milvus', color=MILVUS_COLOR)
    ax2.bar(x + w/2, weav_tput,   w, label='Weaviate', color=WEAVIATE_COLOR)
    ax2.set_ylabel('Throughput (vectors/sec)')
    ax2.set_title('(b) Loading Throughput')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'fig1_loading_performance')

def fig_query_latency(results: Dict[str, Dict], output_dir: Path):
    datasets = [d for d in DISPLAY_ORDER if d in results]
    labels = [DATASET_META[d]['label'] for d in datasets]
    x = np.arange(len(datasets))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, k_key, title in [(ax1, 'k10', '(a) K=10'), (ax2, 'k100', '(b) K=100')]:
        m_p50  = [_get_agg(results[d], k_key, 'P50_mean') for d in datasets]
        m_std  = [_get_agg(results[d], k_key, 'P50_std')  for d in datasets]
        w_p50  = [_get_agg(results[d], k_key, 'P50_mean') for d in datasets]
        w_std  = [_get_agg(results[d], k_key, 'P50_std')  for d in datasets]

        m_p50, m_std, w_p50, w_std = [], [], [], []
        for d in datasets:
            for entry in results[d].get('aggregated', []):
                if entry['K'] == k_key and entry['Database'] == 'Milvus':
                    m_p50.append(entry['P50_mean']); m_std.append(entry['P50_std'])
                elif entry['K'] == k_key and entry['Database'] == 'Weaviate':
                    w_p50.append(entry['P50_mean']); w_std.append(entry['P50_std'])

        m_err_lo = [min(s, v) for s, v in zip(m_std, m_p50)]
        m_err_hi = list(m_std)
        w_err_lo = [min(s, v) for s, v in zip(w_std, w_p50)]
        w_err_hi = list(w_std)

        ax.bar(x - w/2, m_p50, w, yerr=[m_err_lo, m_err_hi], label='Milvus',
               color=MILVUS_COLOR, capsize=3)
        ax.bar(x + w/2, w_p50, w, yerr=[w_err_lo, w_err_hi], label='Weaviate',
               color=WEAVIATE_COLOR, capsize=3)
        ax.set_ylabel('P50 Latency (ms)')
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'fig2_query_latency')

def fig_filter_impact(results: Dict[str, Dict], output_dir: Path):
    datasets = [d for d in DISPLAY_ORDER if d in results]
    labels = [DATASET_META[d]['label'] for d in datasets]
    x = np.arange(len(datasets))
    w = 0.2

    fig, ax = plt.subplots(figsize=(15, 6))

    series = {
        'Milvus':           ('k10',        'Milvus',   MILVUS_COLOR,   1.0),
        'Milvus+filter':    ('k10_filter', 'Milvus',   MILVUS_COLOR,   0.5),
        'Weaviate':         ('k10',        'Weaviate', WEAVIATE_COLOR,  1.0),
        'Weaviate+filter':  ('k10_filter', 'Weaviate', WEAVIATE_COLOR,  0.5),
    }

    for i, (label, (k_key, db, color, alpha)) in enumerate(series.items()):
        vals = []
        for d in datasets:
            v = 0
            for entry in results[d].get('aggregated', []):
                if entry['K'] == k_key and entry['Database'] == db:
                    v = entry['P50_mean']
            vals.append(v)
        offset = (i - 1.5) * w
        ax.bar(x + offset, vals, w, label=label, color=color, alpha=alpha, edgecolor='white')

    ax.set_ylabel('P50 Latency (ms)')
    ax.set_title('Query Latency: No Filter vs With Filter (K=10)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(ncol=2); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'fig3_filter_impact')

def fig_recall(results: Dict[str, Dict], output_dir: Path):
    datasets = [d for d in DISPLAY_ORDER if d in results and results[d].get('recall')]
    labels = [DATASET_META[d]['label'] for d in datasets]
    x = np.arange(len(datasets))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, k_str, title in [(ax1, 'k10', '(a) Recall@10'), (ax2, 'k100', '(b) Recall@100')]:
        m_vals = [results[d]['recall'].get(k_str, {}).get('Milvus', 0) for d in datasets]
        w_vals = [results[d]['recall'].get(k_str, {}).get('Weaviate', 0) for d in datasets]

        bars1 = ax.bar(x - w/2, m_vals, w, label='Milvus', color=MILVUS_COLOR)
        bars2 = ax.bar(x + w/2, w_vals, w, label='Weaviate', color=WEAVIATE_COLOR)
        ax.set_ylabel('Recall')
        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    _save(fig, output_dir, 'fig4_recall')

def fig_concurrent(results: Dict[str, Dict], output_dir: Path):
    # Pick 3 representative datasets
    candidates = ['sift1m', 'gist1m', 'deep-image-96-5M']
    datasets = [d for d in candidates if d in results and results[d].get('concurrent')]

    if not datasets:
        print("  [SKIP] No concurrent data found")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        conc = results[ds]['concurrent']
        for db in ['milvus', 'weaviate']:
            entries = conc.get(db, [])
            if not entries:
                continue
            clients = [e['n_clients'] for e in entries]
            qps     = [e['qps'] for e in entries]
            ax.plot(clients, qps, marker=MARKERS[db.capitalize()],
                    color=COLORS[db.capitalize()], label=db.capitalize(),
                    linewidth=2, markersize=8)

        ax.set_xlabel('Concurrent Clients')
        ax.set_ylabel('Throughput (QPS)')
        ax.set_title(DATASET_META[ds]['label'].replace('\n', ' '))
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 16])

    plt.tight_layout()
    _save(fig, output_dir, 'fig5_concurrent_scaling')

def fig_dimensionality(results: Dict[str, Dict], output_dir: Path):
    # Use datasets of similar size (~1M-1.2M) to isolate dimensionality effect
    # Exclude fashion-mnist (60K), nytimes (290K), deep-image (2M/5M)
    dim_datasets = ['glove-25', 'glove-100', 'sift1m', 'glove-200', 'gist1m']
    dim_datasets = [d for d in dim_datasets if d in results]

    dims = [DATASET_META[d]['dim'] for d in dim_datasets]
    sort_idx = np.argsort(dims)
    dim_datasets = [dim_datasets[i] for i in sort_idx]
    dims = [dims[i] for i in sort_idx]
    dim_labels = [DATASET_META[d]['label'].split('\n')[0] for d in dim_datasets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, k_key, title in [(ax1, 'k10', '(a) K=10'), (ax2, 'k100', '(b) K=100')]:
        for db in ['Milvus', 'Weaviate']:
            vals = []
            for d in dim_datasets:
                for entry in results[d].get('aggregated', []):
                    if entry['K'] == k_key and entry['Database'] == db:
                        vals.append(entry['P50_mean'])
            ax.plot(dims, vals, marker=MARKERS[db], color=COLORS[db],
                    label=db, linewidth=2, markersize=8)

        ax.set_xticks(dims)
        ax.set_xticklabels([f'{d}\n({l})' for d, l in zip(dims, dim_labels)],
                           fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('P50 Latency (ms)')
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Latency vs Dimensionality (datasets ~1M vectors)', fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, 'fig6_dimensionality_impact')

def fig_scale(results: Dict[str, Dict], output_dir: Path):
    # Use datasets with similar low dimensionality to isolate scale effect
    # deep-image-96 (2M, 5M) + sift1m (1M, 128D) + fashion-mnist (60K, 784D)
    # Best isolation: same dataset at different sizes = deep-image 2M vs 5M
    # Also include sift1m (1M) for a third scale point with similar dim
    scale_datasets = ['fashion-mnist-784', 'sift1m', 'deep-image-96-2M', 'deep-image-96-5M']
    scale_datasets = [d for d in scale_datasets if d in results]
    sizes = [DATASET_META[d]['vectors'] for d in scale_datasets]
    sort_idx = np.argsort(sizes)
    scale_datasets = [scale_datasets[i] for i in sort_idx]
    sizes = [sizes[i] for i in sort_idx]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for db in ['Milvus', 'Weaviate']:
        p50_vals, qps_vals = [], []
        for d in scale_datasets:
            found = False
            for entry in results[d].get('aggregated', []):
                if entry['K'] == 'k10' and entry['Database'] == db:
                    p50_vals.append(entry['P50_mean'])
                    qps_vals.append(entry['QPS_mean'])
                    found = True
            if not found:
                p50_vals.append(0)
                qps_vals.append(0)

        ax1.plot(sizes, p50_vals, marker=MARKERS[db], color=COLORS[db],
                 label=db, linewidth=2, markersize=8)
        ax2.plot(sizes, qps_vals, marker=MARKERS[db], color=COLORS[db],
                 label=db, linewidth=2, markersize=8)

    ax1.set_xlabel('Dataset Size (vectors)')
    ax1.set_ylabel('P50 Latency (ms)')
    ax1.set_title('(a) Latency vs Scale (K=10)')
    ax1.set_ylim(bottom=0)
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xscale('log')

    ax2.set_xlabel('Dataset Size (vectors)')
    ax2.set_ylabel('Throughput (QPS)')
    ax2.set_title('(b) QPS vs Scale (K=10)')
    ax2.set_ylim(bottom=0)
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    _save(fig, output_dir, 'fig7_scale_impact')

def fig_resources(results: Dict[str, Dict], output_dir: Path):
    datasets = [d for d in DISPLAY_ORDER if d in results]
    labels = [DATASET_META[d]['label'] for d in datasets]
    x = np.arange(len(datasets))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    m_cpu = [results[d]['loading']['Milvus'].get('avg_cpu_percent', 0) for d in datasets]
    w_cpu = [results[d]['loading']['Weaviate'].get('avg_cpu_percent', 0) for d in datasets]

    ax1.bar(x - w/2, m_cpu, w, label='Milvus',  color=MILVUS_COLOR)
    ax1.bar(x + w/2, w_cpu, w, label='Weaviate', color=WEAVIATE_COLOR)
    ax1.set_ylabel('Avg CPU Usage (%)')
    ax1.set_title('(a) CPU During Data Loading')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    m_mem = [results[d]['loading']['Milvus'].get('peak_memory_mb', 0) for d in datasets]
    w_mem = [results[d]['loading']['Weaviate'].get('peak_memory_mb', 0) for d in datasets]

    ax2.bar(x - w/2, m_mem, w, label='Milvus',  color=MILVUS_COLOR)
    ax2.bar(x + w/2, w_mem, w, label='Weaviate', color=WEAVIATE_COLOR)
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('(b) Memory During Data Loading')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'fig8_resource_usage')

def main():
    parser = argparse.ArgumentParser(description='Generate paper charts from benchmark results')
    parser.add_argument('--results-dir', type=str, default='results/paper')
    parser.add_argument('--output-dir', type=str, default='results/charts')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PAPER-QUALITY CHARTS")
    print("=" * 60)

    results = load_curated(results_dir)
    print(f"Loaded {len(results)} curated benchmark results\n")

    if not results:
        print("No results found. Exiting.")
        return

    fig_loading(results, output_dir)
    fig_query_latency(results, output_dir)
    fig_filter_impact(results, output_dir)
    fig_recall(results, output_dir)
    fig_concurrent(results, output_dir)
    fig_dimensionality(results, output_dir)
    fig_scale(results, output_dir)
    fig_resources(results, output_dir)

    print(f"\nAll charts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
