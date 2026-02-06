"""
Streamlit frontend for visualizing and comparing Milvus vs Weaviate benchmark results from the paper.
============================================================

Author: minageus, lolis, mountzouris
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

st.set_page_config(
    page_title="Milvus vs Weaviate Benchmark",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

MILVUS_COLOR = '#2E86AB'
WEAVIATE_COLOR = '#E94F37'
COLORS = {'Milvus': MILVUS_COLOR, 'Weaviate': WEAVIATE_COLOR}

CURATED_FILES = {
    'fashion-mnist-784':  'paper_benchmark_20260201_210904.json',
    'gist1m':             'paper_benchmark_20260201_163717.json',
    'sift1m':             'paper_benchmark_20260201_184237.json',
    'glove-25':           'paper_benchmark_20260201_195809.json',
    'glove-100':          'paper_benchmark_20260201_230242.json',
    'glove-200':          'paper_benchmark_20260202_085252.json',
    'nytimes-256':        'paper_benchmark_20260201_215039.json',
    'deep-image-96-2M':   'paper_benchmark_20260205_143205.json',
    'deep-image-96-5M':   'paper_benchmark_20260205_184103.json',
}

DATASET_META = {
    'fashion-mnist-784':  {'vectors': 60000,   'dim': 784,  'metric': 'L2',     'label': 'F-MNIST 60K x 784'},
    'gist1m':             {'vectors': 1000000, 'dim': 960,  'metric': 'L2',     'label': 'GIST 1M x 960'},
    'sift1m':             {'vectors': 1000000, 'dim': 128,  'metric': 'L2',     'label': 'SIFT 1M x 128'},
    'glove-25':           {'vectors': 1183514, 'dim': 25,   'metric': 'cosine', 'label': 'GloVe-25 1.2M x 25'},
    'glove-100':          {'vectors': 1183514, 'dim': 100,  'metric': 'cosine', 'label': 'GloVe-100 1.2M x 100'},
    'glove-200':          {'vectors': 1183514, 'dim': 200,  'metric': 'cosine', 'label': 'GloVe-200 1.2M x 200'},
    'nytimes-256':        {'vectors': 290000,  'dim': 256,  'metric': 'cosine', 'label': 'NYTimes 290K x 256'},
    'deep-image-96-2M':   {'vectors': 2000000, 'dim': 96,   'metric': 'L2',     'label': 'Deep-96 2M x 96'},
    'deep-image-96-5M':   {'vectors': 5000000, 'dim': 96,   'metric': 'L2',     'label': 'Deep-96 5M x 96'},
}

DISPLAY_ORDER = [
    'fashion-mnist-784', 'nytimes-256', 'glove-25', 'glove-100',
    'glove-200', 'sift1m', 'gist1m', 'deep-image-96-2M', 'deep-image-96-5M',
]


@st.cache_data
def load_all_results() -> dict:
    results_dir = Path('results/paper')
    data = {}
    for ds_name, filename in CURATED_FILES.items():
        path = results_dir / filename
        if path.exists():
            with open(path) as f:
                data[ds_name] = json.load(f)
    return data


def get_agg(result: dict, k_key: str, db: str, field: str):
    for entry in result.get('aggregated', []):
        if entry.get('K') == k_key and entry.get('Database') == db:
            return entry.get(field, 0)
    return 0


st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E86AB 0%, #E94F37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=60, b=40),
)


def _bar_pair(datasets, m_vals, w_vals, title, yaxis, height=420, log_y=False):
    labels = [DATASET_META[d]['label'] for d in datasets]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Milvus', x=labels, y=m_vals, marker_color=MILVUS_COLOR,
                         text=[f"{v:.1f}" for v in m_vals], textposition='outside'))
    fig.add_trace(go.Bar(name='Weaviate', x=labels, y=w_vals, marker_color=WEAVIATE_COLOR,
                         text=[f"{v:.1f}" for v in w_vals], textposition='outside'))
    fig.update_layout(title=title, yaxis_title=yaxis, barmode='group', height=height, **PLOTLY_LAYOUT)
    if log_y:
        fig.update_yaxes(type='log')
    return fig

def chart_loading(results, datasets):
    m_time = [results[d]['loading']['Milvus']['load_time_seconds'] for d in datasets]
    w_time = [results[d]['loading']['Weaviate']['load_time_seconds'] for d in datasets]
    n_vecs = [DATASET_META[d]['vectors'] for d in datasets]
    m_tput = [n / t for n, t in zip(n_vecs, m_time)]
    w_tput = [n / t for n, t in zip(n_vecs, w_time)]

    fig1 = _bar_pair(datasets, m_time, w_time, 'Data Loading Time', 'Time (seconds)', log_y=True)
    fig2 = _bar_pair(datasets, m_tput, w_tput, 'Loading Throughput', 'Vectors / sec')
    return fig1, fig2


def chart_query_latency(results, datasets):
    figs = []
    for k_key, title in [('k10', 'Query Latency K=10'), ('k100', 'Query Latency K=100')]:
        labels = [DATASET_META[d]['label'] for d in datasets]
        m_p50 = [get_agg(results[d], k_key, 'Milvus', 'P50_mean') for d in datasets]
        m_std = [get_agg(results[d], k_key, 'Milvus', 'P50_std') for d in datasets]
        w_p50 = [get_agg(results[d], k_key, 'Weaviate', 'P50_mean') for d in datasets]
        w_std = [get_agg(results[d], k_key, 'Weaviate', 'P50_std') for d in datasets]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Milvus', x=labels, y=m_p50, error_y=dict(type='data', array=m_std),
                             marker_color=MILVUS_COLOR))
        fig.add_trace(go.Bar(name='Weaviate', x=labels, y=w_p50, error_y=dict(type='data', array=w_std),
                             marker_color=WEAVIATE_COLOR))
        fig.update_layout(title=title, yaxis_title='P50 Latency (ms)', barmode='group',
                          height=420, **PLOTLY_LAYOUT)
        fig.update_yaxes(rangemode='tozero')
        figs.append(fig)
    return figs


def chart_filter_impact(results, datasets):
    labels = [DATASET_META[d]['label'] for d in datasets]
    series = [
        ('Milvus',          'k10',        'Milvus',   MILVUS_COLOR,  1.0),
        ('Milvus+filter',   'k10_filter', 'Milvus',   MILVUS_COLOR,  0.5),
        ('Weaviate',        'k10',        'Weaviate', WEAVIATE_COLOR, 1.0),
        ('Weaviate+filter', 'k10_filter', 'Weaviate', WEAVIATE_COLOR, 0.5),
    ]
    fig = go.Figure()
    for label, k_key, db, color, opacity in series:
        vals = [get_agg(results[d], k_key, db, 'P50_mean') for d in datasets]
        fig.add_trace(go.Bar(name=label, x=labels, y=vals, marker_color=color,
                             opacity=opacity,
                             text=[f"{v:.1f}" for v in vals], textposition='outside'))
    fig.update_layout(title='Filter Impact on Latency (K=10)', yaxis_title='P50 Latency (ms)',
                      barmode='group', height=450, **PLOTLY_LAYOUT)
    fig.update_yaxes(rangemode='tozero')
    return fig


def chart_recall(results, datasets):
    datasets_with_recall = [d for d in datasets if results[d].get('recall')]
    if not datasets_with_recall:
        return None, None

    figs = []
    for k_str, title in [('k10', 'Recall@10'), ('k100', 'Recall@100')]:
        labels = [DATASET_META[d]['label'] for d in datasets_with_recall]
        m_vals = [results[d]['recall'].get(k_str, {}).get('Milvus', 0) for d in datasets_with_recall]
        w_vals = [results[d]['recall'].get(k_str, {}).get('Weaviate', 0) for d in datasets_with_recall]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Milvus', x=labels, y=m_vals, marker_color=MILVUS_COLOR,
                             text=[f"{v:.2f}" for v in m_vals], textposition='outside'))
        fig.add_trace(go.Bar(name='Weaviate', x=labels, y=w_vals, marker_color=WEAVIATE_COLOR,
                             text=[f"{v:.2f}" for v in w_vals], textposition='outside'))
        fig.update_layout(title=title, yaxis_title='Recall', barmode='group', height=420, **PLOTLY_LAYOUT)
        fig.update_yaxes(range=[0, 1.08])
        figs.append(fig)
    return figs


def chart_concurrent(results, datasets):
    candidates = [d for d in datasets if results[d].get('concurrent')]
    if not candidates:
        return None

    fig = make_subplots(rows=1, cols=len(candidates),
                        subplot_titles=[DATASET_META[d]['label'] for d in candidates],
                        horizontal_spacing=0.08)

    for col, ds in enumerate(candidates, 1):
        conc = results[ds]['concurrent']
        for db_key, db_name in [('milvus', 'Milvus'), ('weaviate', 'Weaviate')]:
            entries = conc.get(db_key, [])
            if not entries:
                continue
            clients = [e['n_clients'] for e in entries]
            qps = [e['qps'] for e in entries]
            fig.add_trace(go.Scatter(
                x=clients, y=qps, mode='lines+markers', name=db_name,
                line=dict(color=COLORS[db_name], width=2),
                marker=dict(size=8),
                showlegend=(col == 1),
            ), row=1, col=col)
        fig.update_xaxes(title_text='Clients', row=1, col=col)
        fig.update_yaxes(title_text='QPS' if col == 1 else '', row=1, col=col)

    fig.update_layout(title='Concurrent Scaling: QPS vs Client Count',
                      height=400, **PLOTLY_LAYOUT)
    return fig


def chart_dimensionality(results, datasets):
    dim_ds = ['glove-25', 'glove-100', 'sift1m', 'glove-200', 'gist1m']
    dim_ds = [d for d in dim_ds if d in results]
    dims = sorted([(DATASET_META[d]['dim'], d) for d in dim_ds])
    dim_ds = [d for _, d in dims]
    dim_vals = [DATASET_META[d]['dim'] for d in dim_ds]
    dim_labels = [DATASET_META[d]['label'] for d in dim_ds]

    fig = make_subplots(rows=1, cols=2, subplot_titles=['K=10', 'K=100'], horizontal_spacing=0.1)

    for col, k_key in enumerate(['k10', 'k100'], 1):
        for db in ['Milvus', 'Weaviate']:
            vals = [get_agg(results[d], k_key, db, 'P50_mean') for d in dim_ds]
            fig.add_trace(go.Scatter(
                x=dim_vals, y=vals, mode='lines+markers', name=db,
                line=dict(color=COLORS[db], width=2), marker=dict(size=8),
                showlegend=(col == 1),
            ), row=1, col=col)
        fig.update_xaxes(title_text='Dimensionality', row=1, col=col,
                         tickvals=dim_vals, ticktext=dim_labels)
        fig.update_yaxes(title_text='P50 Latency (ms)' if col == 1 else '', row=1, col=col,
                         rangemode='tozero')

    fig.update_layout(title='Latency vs Dimensionality (datasets ~1M vectors)',
                      height=420, **PLOTLY_LAYOUT)
    return fig


def chart_scale(results, datasets):
    scale_ds = ['fashion-mnist-784', 'sift1m', 'deep-image-96-2M', 'deep-image-96-5M']
    scale_ds = [d for d in scale_ds if d in results]
    sizes = [DATASET_META[d]['vectors'] for d in scale_ds]
    scale_labels = [DATASET_META[d]['label'] for d in scale_ds]

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Latency vs Scale (K=10)', 'QPS vs Scale (K=10)'],
                        horizontal_spacing=0.1)

    for db in ['Milvus', 'Weaviate']:
        p50 = [get_agg(results[d], 'k10', db, 'P50_mean') for d in scale_ds]
        qps = [get_agg(results[d], 'k10', db, 'QPS_mean') for d in scale_ds]

        fig.add_trace(go.Scatter(
            x=sizes, y=p50, mode='lines+markers', name=db,
            line=dict(color=COLORS[db], width=2), marker=dict(size=8),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sizes, y=qps, mode='lines+markers', name=db,
            line=dict(color=COLORS[db], width=2), marker=dict(size=8),
            showlegend=False,
        ), row=1, col=2)

    for col in [1, 2]:
        fig.update_xaxes(type='log', title_text='Dataset Size', row=1, col=col,
                         tickvals=sizes, ticktext=scale_labels)
    fig.update_yaxes(title_text='P50 (ms)', rangemode='tozero', row=1, col=1)
    fig.update_yaxes(title_text='QPS', rangemode='tozero', row=1, col=2)

    fig.update_layout(title='Performance vs Dataset Scale', height=420, **PLOTLY_LAYOUT)
    return fig


def chart_resources(results, datasets):
    m_cpu = [results[d]['loading']['Milvus'].get('avg_cpu_percent', 0) for d in datasets]
    w_cpu = [results[d]['loading']['Weaviate'].get('avg_cpu_percent', 0) for d in datasets]
    m_mem = [results[d]['loading']['Milvus'].get('peak_memory_mb', 0) for d in datasets]
    w_mem = [results[d]['loading']['Weaviate'].get('peak_memory_mb', 0) for d in datasets]

    fig1 = _bar_pair(datasets, m_cpu, w_cpu, 'Avg CPU During Loading', 'CPU Usage (%)')
    fig2 = _bar_pair(datasets, m_mem, w_mem, 'Peak Memory During Loading', 'Memory (MB)')
    return fig1, fig2

def single_loading_chart(data, ds_name):
    m = data['loading']['Milvus']
    w = data['loading']['Weaviate']
    metrics = ['Load Time (s)', 'Avg CPU (%)', 'Peak Memory (MB)', 'Disk Write (MB)']
    m_vals = [m['load_time_seconds'], m.get('avg_cpu_percent', 0),
              m.get('peak_memory_mb', 0), m.get('disk_write_mb', 0)]
    w_vals = [w['load_time_seconds'], w.get('avg_cpu_percent', 0),
              w.get('peak_memory_mb', 0), w.get('disk_write_mb', 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Milvus', y=metrics, x=m_vals, orientation='h',
                         marker_color=MILVUS_COLOR,
                         text=[f"{v:.1f}" for v in m_vals], textposition='outside'))
    fig.add_trace(go.Bar(name='Weaviate', y=metrics, x=w_vals, orientation='h',
                         marker_color=WEAVIATE_COLOR,
                         text=[f"{v:.1f}" for v in w_vals], textposition='outside'))
    fig.update_layout(title=f'Loading Comparison: {DATASET_META[ds_name]["label"]}',
                      barmode='group', height=350, **PLOTLY_LAYOUT)
    return fig


def single_latency_chart(data):
    fig = go.Figure()
    for k_key in ['k10', 'k100']:
        for db in ['Milvus', 'Weaviate']:
            for pct, dash in [('P50_mean', 'solid'), ('P95_mean', 'dash')]:
                val = get_agg(data, k_key, db, pct)
                label = f"{db} {pct.replace('_mean','')} ({k_key})"
                fig.add_trace(go.Bar(name=label, x=[f"{k_key.upper()} {pct.replace('_mean','')}"],
                                     y=[val], marker_color=COLORS[db],
                                     opacity=1.0 if 'P50' in pct else 0.6,
                                     text=[f"{val:.2f}"], textposition='outside'))
    fig.update_layout(title='Query Latency Breakdown', yaxis_title='Latency (ms)',
                      barmode='group', height=400, **PLOTLY_LAYOUT)
    fig.update_yaxes(rangemode='tozero')
    return fig


def single_filter_chart(data):
    fig = go.Figure()
    for k_key_base in ['k10', 'k100']:
        for db in ['Milvus', 'Weaviate']:
            no_filt = get_agg(data, k_key_base, db, 'P50_mean')
            filt = get_agg(data, f'{k_key_base}_filter', db, 'P50_mean')
            fig.add_trace(go.Bar(
                name=f'{db} {k_key_base.upper()}',
                x=['No Filter', 'With Filter'],
                y=[no_filt, filt],
                marker_color=COLORS[db],
                text=[f"{no_filt:.2f}", f"{filt:.2f}"],
                textposition='outside',
            ))
    fig.update_layout(title='Filter Impact', yaxis_title='P50 Latency (ms)',
                      barmode='group', height=380, **PLOTLY_LAYOUT)
    fig.update_yaxes(rangemode='tozero')
    return fig


def single_concurrent_chart(data, ds_name):
    conc = data.get('concurrent')
    if not conc:
        return None
    fig = go.Figure()
    for db_key, db_name in [('milvus', 'Milvus'), ('weaviate', 'Weaviate')]:
        entries = conc.get(db_key, [])
        if not entries:
            continue
        clients = [e['n_clients'] for e in entries]
        qps = [e['qps'] for e in entries]
        p50 = [e['p50_ms'] for e in entries]
        fig.add_trace(go.Scatter(
            x=clients, y=qps, mode='lines+markers', name=f'{db_name} QPS',
            line=dict(color=COLORS[db_name], width=2), marker=dict(size=8),
            text=[f"P50={p:.1f}ms" for p in p50], hoverinfo='text+x+y',
        ))
    fig.update_layout(title=f'Concurrent Scaling: {DATASET_META[ds_name]["label"]}',
                      xaxis_title='Concurrent Clients', yaxis_title='QPS',
                      height=400, **PLOTLY_LAYOUT)
    fig.update_xaxes(tickvals=[1, 2, 4, 8, 16])
    return fig

def build_summary_table(results, datasets):
    rows = []
    for d in datasets:
        r = results[d]
        meta = DATASET_META[d]
        m_load = r['loading']['Milvus']['load_time_seconds']
        w_load = r['loading']['Weaviate']['load_time_seconds']
        m_p50 = get_agg(r, 'k10', 'Milvus', 'P50_mean')
        w_p50 = get_agg(r, 'k10', 'Weaviate', 'P50_mean')
        m_qps = get_agg(r, 'k10', 'Milvus', 'QPS_mean')
        w_qps = get_agg(r, 'k10', 'Weaviate', 'QPS_mean')
        m_recall = r.get('recall', {}).get('k10', {}).get('Milvus', 0)
        w_recall = r.get('recall', {}).get('k10', {}).get('Weaviate', 0)

        rows.append({
            'Dataset': meta['label'],
            'Vectors': f"{meta['vectors']:,}",
            'Dim': meta['dim'],
            'Metric': meta['metric'],
            'M Load (s)': f"{m_load:.1f}",
            'W Load (s)': f"{w_load:.1f}",
            'M P50 (ms)': f"{m_p50:.2f}",
            'W P50 (ms)': f"{w_p50:.2f}",
            'M QPS': f"{m_qps:.0f}",
            'W QPS': f"{w_qps:.0f}",
            'M Recall@10': f"{m_recall:.2%}",
            'W Recall@10': f"{w_recall:.2%}",
        })
    return pd.DataFrame(rows)

def main():
    st.markdown('<h1 class="main-header">Vector Database Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Milvus vs Weaviate &mdash; Paper Results Dashboard</p>',
                unsafe_allow_html=True)

    results = load_all_results()
    if not results:
        st.error("No results found in `results/paper/`. Run benchmarks first.")
        return

    available = [d for d in DISPLAY_ORDER if d in results]

    st.sidebar.header("Navigation")
    view = st.sidebar.radio("View", ["Cross-Dataset Analysis", "Single Dataset Deep Dive"])

    if view == "Cross-Dataset Analysis":
        st.sidebar.subheader("Datasets")
        selected = st.sidebar.multiselect(
            "Include datasets:", available, default=available,
            format_func=lambda d: DATASET_META[d]['label'],
        )
        if not selected:
            st.warning("Select at least one dataset.")
            return

        datasets = [d for d in DISPLAY_ORDER if d in selected]

        st.header("Summary")
        st.dataframe(build_summary_table(results, datasets), use_container_width=True, hide_index=True)
        st.divider()

        st.header("Loading Performance")
        fig1, fig2 = chart_loading(results, datasets)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)
        st.divider()

        st.header("Query Latency")
        figs = chart_query_latency(results, datasets)
        c1, c2 = st.columns(2)
        c1.plotly_chart(figs[0], use_container_width=True)
        c2.plotly_chart(figs[1], use_container_width=True)
        st.divider()

        st.header("Filter Impact")
        fig = chart_filter_impact(results, datasets)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        st.header("Recall@K Accuracy")
        recall_figs = chart_recall(results, datasets)
        if recall_figs:
            c1, c2 = st.columns(2)
            c1.plotly_chart(recall_figs[0], use_container_width=True)
            c2.plotly_chart(recall_figs[1], use_container_width=True)
        st.divider()

        st.header("Concurrent Scaling")
        conc_fig = chart_concurrent(results, datasets)
        if conc_fig:
            st.plotly_chart(conc_fig, use_container_width=True)
        else:
            st.info("No concurrent data available for the selected datasets.")
        st.divider()

        st.header("Dimensionality Impact")
        dim_fig = chart_dimensionality(results, datasets)
        st.plotly_chart(dim_fig, use_container_width=True)
        st.caption("Using datasets of ~1M vectors (GloVe-25/100/200, SIFT, GIST) to isolate dimensionality effect.")
        st.divider()

        st.header("Scale Impact")
        scale_fig = chart_scale(results, datasets)
        st.plotly_chart(scale_fig, use_container_width=True)
        st.caption("Using F-MNIST (60K), SIFT (1M), Deep-96 (2M, 5M) to show scale effect.")
        st.divider()

        st.header("Resource Usage During Loading")
        rfig1, rfig2 = chart_resources(results, datasets)
        c1, c2 = st.columns(2)
        c1.plotly_chart(rfig1, use_container_width=True)
        c2.plotly_chart(rfig2, use_container_width=True)

    else:
        ds_name = st.sidebar.selectbox(
            "Select dataset:", available,
            format_func=lambda d: DATASET_META[d]['label'],
        )
        data = results[ds_name]
        meta = DATASET_META[ds_name]

        st.header(meta['label'])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vectors", f"{meta['vectors']:,}")
        c2.metric("Dimensions", meta['dim'])
        c3.metric("Distance Metric", meta['metric'])
        c4.metric("Index", data.get('config', {}).get('index_config', {}).get('M', '16'))
        st.divider()

        st.subheader("Loading Performance")
        m = data['loading']['Milvus']
        w = data['loading']['Weaviate']
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Milvus Load Time", f"{m['load_time_seconds']:.1f}s")
            st.metric("Milvus Peak Memory", f"{m.get('peak_memory_mb', 0):.0f} MB")
            st.metric("Milvus Avg CPU", f"{m.get('avg_cpu_percent', 0):.1f}%")
        with c2:
            st.metric("Weaviate Load Time", f"{w['load_time_seconds']:.1f}s")
            st.metric("Weaviate Peak Memory", f"{w.get('peak_memory_mb', 0):.0f} MB")
            st.metric("Weaviate Avg CPU", f"{w.get('avg_cpu_percent', 0):.1f}%")

        fig = single_loading_chart(data, ds_name)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        st.subheader("Query Latency")
        perf_rows = []
        for entry in data.get('aggregated', []):
            perf_rows.append({
                'Database': entry['Database'],
                'K': entry['K'],
                'P50 (ms)': entry.get('P50 (ms)', ''),
                'P95 (ms)': entry.get('P95 (ms)', ''),
                'QPS': entry.get('QPS', ''),
            })
        if perf_rows:
            st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

        fig = single_latency_chart(data)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        st.subheader("Filter Impact")
        fig = single_filter_chart(data)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        if data.get('recall'):
            st.subheader("Recall@K")
            recall = data['recall']
            rc1, rc2 = st.columns(2)
            for col, k_str in [(rc1, 'k10'), (rc2, 'k100')]:
                vals = recall.get(k_str, {})
                with col:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=['Milvus', 'Weaviate'],
                                        y=[vals.get('Milvus', 0), vals.get('Weaviate', 0)],
                                        marker_color=[MILVUS_COLOR, WEAVIATE_COLOR],
                                        text=[f"{vals.get('Milvus',0):.4f}", f"{vals.get('Weaviate',0):.4f}"],
                                        textposition='outside'))
                    fig.update_layout(title=f'Recall@{k_str.replace("k","")}', yaxis_title='Recall',
                                      height=350, **PLOTLY_LAYOUT)
                    fig.update_yaxes(range=[0, 1.08])
                    st.plotly_chart(fig, use_container_width=True)
            st.divider()

        if data.get('concurrent'):
            st.subheader("Concurrent Scaling")
            fig = single_concurrent_chart(data, ds_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

                conc_rows = []
                for db_key in ['milvus', 'weaviate']:
                    for e in data['concurrent'].get(db_key, []):
                        conc_rows.append({
                            'Database': db_key.capitalize(),
                            'Clients': e['n_clients'],
                            'QPS': f"{e['qps']:.1f}",
                            'P50 (ms)': f"{e['p50_ms']:.2f}",
                            'P95 (ms)': f"{e['p95_ms']:.2f}",
                            'P99 (ms)': f"{e['p99_ms']:.2f}",
                            'Failed': e.get('failed', 0),
                        })
                st.dataframe(pd.DataFrame(conc_rows), use_container_width=True, hide_index=True)
            st.divider()

        if data.get('query_resources'):
            st.subheader("Resource Usage During Queries")
            qr = data['query_resources']
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg CPU", f"{qr.get('avg_cpu_percent', 0):.1f}%")
            c2.metric("Peak Memory", f"{qr.get('peak_memory_mb', 0):.0f} MB")
            c3.metric("Disk Write", f"{qr.get('disk_write_mb', 0):.0f} MB")
            st.divider()

        with st.expander("Raw JSON Data"):
            st.json(data)

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#888;'>"
        "Built by minageus, lolis, mountzouris "
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()