"""
Streamlit Dashboard: Milvus vs Weaviate Benchmark Results
=========================================================

Interactive visualization of vector database benchmark results.
Loads all complete_benchmark_*.txt files and comparison/recall CSVs.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Milvus vs Weaviate Benchmark",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .winner-milvus {
        color: #00b894;
        font-weight: bold;
    }
    .winner-weaviate {
        color: #e17055;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def get_results_files():
    """Get all result files from the results directory."""
    results_dir = Path('results')
    if not results_dir.exists():
        return [], [], []
    
    comparison_files = sorted(results_dir.glob('comparison_*.csv'), reverse=True)
    recall_files = sorted(results_dir.glob('recall_*.csv'), reverse=True)
    report_files = sorted(results_dir.glob('complete_benchmark_*.txt'), reverse=True)
    
    return comparison_files, recall_files, report_files


def get_ai_recommendations(comparison_df, recall_df, api_key):
    """
    Use OpenAI to analyze benchmark data and provide recommendations.
    
    Args:
        comparison_df: DataFrame with query performance metrics
        recall_df: DataFrame with recall@K accuracy metrics
        api_key: OpenAI API key
    
    Returns:
        str: AI-generated analysis and recommendations
    """
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare data summary for the AI
        data_summary = "## Benchmark Data Summary\n\n"
        
        if comparison_df is not None and not comparison_df.empty:
            data_summary += "### Query Performance Metrics:\n"
            data_summary += comparison_df.to_string(index=False)
            data_summary += "\n\n"
        
        if recall_df is not None and not recall_df.empty:
            data_summary += "### Recall@K Accuracy (Search Quality):\n"
            data_summary += recall_df.to_string(index=False)
            data_summary += "\n\n"
        
        prompt = f"""
        You are an expert in vector databases, specifically Milvus and Weaviate.
        Analyze the following benchmark results comparing these two databases:

        {data_summary}
        
        Based on this data, provide:
        1. A brief analysis of the results (2-3 sentences)
        2. When to choose Milvus (4-5 bullet points based on the actual data)
        3. When to choose Weaviate (4-5 bullet points based on the actual data)
        
        Format your response in Markdown with clear headers.
        Be specific and reference actual numbers from the data where relevant.
        Keep it concise but insightful.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical expert in vector databases and benchmarking. Provide clear, data-driven recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"


def extract_timestamp(filename):
    """Extract timestamp from filename."""
    match = re.search(r'(\d{8}_\d{6})', str(filename))
    if match:
        ts = match.group(1)
        try:
            dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ts
    return str(filename)


def parse_complete_benchmark(report_path):
    """Parse the complete benchmark report file for all data."""
    data = {
        'timestamp': extract_timestamp(report_path),
        'filepath': str(report_path),
        'date': '',
        'dataset': '',
        'vectors': 0,
        'dimensions': 0,
        'milvus_load_time': 0,
        'weaviate_load_time': 0,
        'milvus_peak_memory': 0,
        'weaviate_peak_memory': 0,
        'raw_data_size': 0,
        'query_performance': [],
        'recall': {}
    }
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract dataset info
        dataset_match = re.search(r'Dataset:\s*(\w+)\s*\(([0-9,]+)\s*vectors,\s*(\d+)D\)', content)
        if dataset_match:
            data['dataset'] = dataset_match.group(1)
            data['vectors'] = int(dataset_match.group(2).replace(',', ''))
            data['dimensions'] = int(dataset_match.group(3))
        
        # Extract date
        date_match = re.search(r'Date:\s*(.+)', content)
        if date_match:
            data['date'] = date_match.group(1).strip()
        
        # Extract Milvus stats
        milvus_load_match = re.search(r'Milvus:\s*\n\s*Load Time:\s*([\d.]+)\s*seconds', content)
        if milvus_load_match:
            data['milvus_load_time'] = float(milvus_load_match.group(1))
        
        milvus_mem_match = re.search(r'Milvus:\s*\n\s*Load Time:.*\n\s*Peak Memory:\s*([\d.]+)\s*MB', content)
        if milvus_mem_match:
            data['milvus_peak_memory'] = float(milvus_mem_match.group(1))
        
        # Extract Weaviate stats
        weaviate_load_match = re.search(r'Weaviate:\s*\n\s*Load Time:\s*([\d.]+)\s*seconds', content)
        if weaviate_load_match:
            data['weaviate_load_time'] = float(weaviate_load_match.group(1))
        
        weaviate_mem_match = re.search(r'Weaviate:\s*\n\s*Load Time:.*\n\s*Peak Memory:\s*([\d.]+)\s*MB', content)
        if weaviate_mem_match:
            data['weaviate_peak_memory'] = float(weaviate_mem_match.group(1))
        
        # Extract raw data size
        size_match = re.search(r'Raw Data Size:\s*([\d.]+)\s*MB', content)
        if size_match:
            data['raw_data_size'] = float(size_match.group(1))
        
        # Extract recall values
        recall_pattern = re.findall(r'Recall@(\d+):\s*Milvus=([\d.]+),\s*Weaviate=([\d.]+)', content)
        if recall_pattern:
            for k, m, w in recall_pattern:
                data['recall'][int(k)] = {'Milvus': float(m), 'Weaviate': float(w)}
            
    except Exception as e:
        st.warning(f"Could not parse report file {report_path}: {e}")
    
    return data


def load_all_benchmarks(report_files):
    """Load all benchmark reports and return a list of parsed data."""
    benchmarks = []
    for report_file in report_files:
        data = parse_complete_benchmark(report_file)
        benchmarks.append(data)
    return benchmarks


def load_comparison_data(file_path):
    """Load and process comparison CSV data."""
    df = pd.read_csv(file_path)
    return df


def load_recall_data(file_path):
    """Load and process recall CSV data."""
    df = pd.read_csv(file_path)
    return df


def create_latency_chart(df, metric='P50 (ms)'):
    """Create latency comparison bar chart."""
    fig = px.bar(
        df,
        x='Test',
        y=metric,
        color='Database',
        barmode='group',
        title=f'Query Latency Comparison ({metric})',
        color_discrete_map={'Milvus': '#00b894', 'Weaviate': '#e17055'}
    )
    
    fig.update_layout(
        xaxis_title='Test Configuration',
        yaxis_title=f'Latency ({metric})',
        legend_title='Database',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_qps_chart(df):
    """Create QPS comparison bar chart."""
    fig = px.bar(
        df,
        x='Test',
        y='QPS',
        color='Database',
        barmode='group',
        title='Queries Per Second (QPS) Comparison',
        color_discrete_map={'Milvus': '#00b894', 'Weaviate': '#e17055'}
    )
    
    fig.update_layout(
        xaxis_title='Test Configuration',
        yaxis_title='Queries Per Second',
        legend_title='Database',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_recall_chart(recall_data):
    """Create recall accuracy line chart from parsed data."""
    if not recall_data:
        return None
    
    k_values = sorted(recall_data.keys())
    milvus_vals = [recall_data[k]['Milvus'] for k in k_values]
    weaviate_vals = [recall_data[k]['Weaviate'] for k in k_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_values, y=milvus_vals, name='Milvus',
        mode='lines+markers', line=dict(color='#00b894', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=k_values, y=weaviate_vals, name='Weaviate',
        mode='lines+markers', line=dict(color='#e17055', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Recall@K Accuracy Comparison',
        xaxis_title='K (Number of Results)',
        yaxis_title='Recall Score',
        yaxis_range=[0, 1.05],
        legend_title='Database',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_recall_chart_from_df(df):
    """Create recall accuracy line chart from dataframe."""
    df_melted = df.melt(id_vars=['K'], var_name='Database', value_name='Recall')
    
    fig = px.line(
        df_melted,
        x='K',
        y='Recall',
        color='Database',
        markers=True,
        title='Recall@K Accuracy Comparison',
        color_discrete_map={'Milvus': '#00b894', 'Weaviate': '#e17055'}
    )
    
    fig.update_layout(
        xaxis_title='K (Number of Results)',
        yaxis_title='Recall Score',
        yaxis_range=[0, 1.05],
        legend_title='Database',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_traces(line_width=3, marker_size=10)
    
    return fig


def create_multi_run_comparison(benchmarks, metric='load_time'):
    """Create comparison chart across multiple benchmark runs."""
    if not benchmarks:
        return None
    
    labels = []
    milvus_values = []
    weaviate_values = []
    
    for bench in benchmarks:
        label = f"{bench['dataset']}\n({bench['timestamp'][:10]})"
        labels.append(label)
        
        if metric == 'load_time':
            milvus_values.append(bench['milvus_load_time'])
            weaviate_values.append(bench['weaviate_load_time'])
        elif metric == 'memory':
            milvus_values.append(bench['milvus_peak_memory'])
            weaviate_values.append(bench['weaviate_peak_memory'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Milvus', x=labels, y=milvus_values,
        marker_color='#00b894'
    ))
    fig.add_trace(go.Bar(
        name='Weaviate', x=labels, y=weaviate_values,
        marker_color='#e17055'
    ))
    
    title_map = {
        'load_time': 'Load Time Comparison Across Runs (seconds)',
        'memory': 'Peak Memory Usage Across Runs (MB)'
    }
    
    fig.update_layout(
        title=title_map.get(metric, 'Comparison'),
        barmode='group',
        template='plotly_white',
        xaxis_title='Benchmark Run',
        yaxis_title='Seconds' if metric == 'load_time' else 'MB'
    )
    
    return fig


def create_summary_table(benchmarks):
    """Create a summary table of all benchmark runs."""
    rows = []
    for bench in benchmarks:
        load_winner = 'Milvus' if bench['milvus_load_time'] < bench['weaviate_load_time'] else 'Weaviate'
        mem_winner = 'Milvus' if bench['milvus_peak_memory'] < bench['weaviate_peak_memory'] else 'Weaviate'
        
        # Check recall winner (recall@10 if available)
        recall_winner = 'N/A'
        recall_10_m = 0
        recall_10_w = 0
        if 10 in bench['recall']:
            recall_10_m = bench['recall'][10]['Milvus']
            recall_10_w = bench['recall'][10]['Weaviate']
            recall_winner = 'Milvus' if recall_10_m > recall_10_w else 'Weaviate'
        
        rows.append({
            'Run Date': bench['timestamp'],
            'Dataset': bench['dataset'],
            'Vectors': f"{bench['vectors']:,}",
            'Dimensions': bench['dimensions'],
            'Milvus Load (s)': f"{bench['milvus_load_time']:.1f}",
            'Weaviate Load (s)': f"{bench['weaviate_load_time']:.1f}",
            'Load Winner': load_winner,
            'Milvus Recall@10': f"{recall_10_m:.4f}" if recall_10_m else 'N/A',
            'Weaviate Recall@10': f"{recall_10_w:.4f}" if recall_10_w else 'N/A',
            'Recall Winner': recall_winner
        })
    
    return pd.DataFrame(rows)


def main():
    # Header
    st.markdown('<h1 class="main-header">Milvus vs Weaviate Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive visualization of vector database performance comparison</p>', unsafe_allow_html=True)
    
    # Get available result files
    comparison_files, recall_files, report_files = get_results_files()
    
    if not report_files:
        st.error("No benchmark results found in the 'results/' directory.")
        st.info("Run `python run_benchmark.py` to generate benchmark results first.")
        return
    
    # Load all benchmark data from complete_benchmark_*.txt files
    all_benchmarks = load_all_benchmarks(report_files)
    
    # Sidebar for navigation and options
    with st.sidebar:
        st.header("📁 Navigation")
        
        view_mode = st.radio(
            "View Mode",
            ["Single Run Analysis", "Multi-Run Comparison"],
            index=0
        )
        
        if view_mode == "Single Run Analysis":
            # Create options with readable labels
            run_options = {f"{b['dataset']} ({b['timestamp']})": i for i, b in enumerate(all_benchmarks)}
            selected_label = st.selectbox(
                "Select Benchmark Run",
                options=list(run_options.keys()),
                index=0
            )
            selected_idx = run_options[selected_label]
        else:
            selected_runs = st.multiselect(
                "Select Runs to Compare",
                options=[f"{b['dataset']} ({b['timestamp']})" for b in all_benchmarks],
                default=[f"{b['dataset']} ({b['timestamp']})" for b in all_benchmarks]
            )
        
        st.divider()
        st.header("Display Options")
        
        show_loading = st.checkbox("Show Loading Performance", value=True)
        show_latency = st.checkbox("Show Query Latency", value=True)
        show_qps = st.checkbox("Show QPS Comparison", value=True)
        show_recall = st.checkbox("Show Recall Accuracy", value=True)
        show_raw_data = st.checkbox("Show Raw Data Tables", value=False)
    
    # =========================================================================
    # MULTI-RUN COMPARISON VIEW
    # =========================================================================
    if view_mode == "Multi-Run Comparison":
        st.header("Multi-Run Benchmark Comparison")
        
        st.info(f"Loaded **{len(all_benchmarks)}** benchmark runs from `results/` directory")
        
        # Summary Table
        st.subheader("All Benchmark Runs Summary")
        summary_df = create_summary_table(all_benchmarks)
        st.dataframe(summary_df, use_container_width=True)
        
        st.divider()
        
        # Filter benchmarks based on selection
        if selected_runs:
            filtered_benchmarks = [b for b in all_benchmarks 
                                   if f"{b['dataset']} ({b['timestamp']})" in selected_runs]
        else:
            filtered_benchmarks = all_benchmarks
        
        if show_loading:
            st.subheader("Load Time Across Runs")
            fig = create_multi_run_comparison(filtered_benchmarks, 'load_time')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Peak Memory Usage Across Runs")
            fig = create_multi_run_comparison(filtered_benchmarks, 'memory')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        if show_recall:
            st.subheader("Recall@K Comparison Across Runs")
            
            # Create recall comparison for all runs
            recall_data = []
            for bench in filtered_benchmarks:
                if bench['recall']:
                    for k, vals in bench['recall'].items():
                        recall_data.append({
                            'Run': f"{bench['dataset']} ({bench['timestamp'][:10]})",
                            'K': k,
                            'Milvus': vals['Milvus'],
                            'Weaviate': vals['Weaviate']
                        })
            
            if recall_data:
                recall_df = pd.DataFrame(recall_data)
                
                # Milvus chart
                fig_milvus = px.line(
                    recall_df, x='K', y='Milvus', color='Run',
                    markers=True, title='Milvus Recall@K Across Runs'
                )
                fig_milvus.update_layout(template='plotly_white', yaxis_range=[0, 1.05])
                
                # Weaviate chart
                fig_weaviate = px.line(
                    recall_df, x='K', y='Weaviate', color='Run',
                    markers=True, title='Weaviate Recall@K Across Runs'
                )
                fig_weaviate.update_layout(template='plotly_white', yaxis_range=[0, 1.05])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_milvus, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_weaviate, use_container_width=True)
    
    # =========================================================================
    # SINGLE RUN ANALYSIS VIEW
    # =========================================================================
    else:
        selected_bench = all_benchmarks[selected_idx]
        
        # Display benchmark info
        st.info(f"**Dataset**: {selected_bench['dataset']} | **Vectors**: {selected_bench['vectors']:,} | **Dimensions**: {selected_bench['dimensions']}D | **Raw Size**: {selected_bench.get('raw_data_size', 'N/A')} MB")
        
        # Find matching CSV files
        ts_match = re.search(r'(\d{8}_\d{6})', selected_bench['filepath'])
        comparison_df = None
        recall_df = None
        
        if ts_match:
            ts = ts_match.group(1)
            matching_comparison = [f for f in comparison_files if ts in str(f)]
            matching_recall = [f for f in recall_files if ts in str(f)]
            
            if matching_comparison:
                comparison_df = load_comparison_data(matching_comparison[0])
            if matching_recall:
                recall_df = load_recall_data(matching_recall[0])
        
        # Key Metrics Summary
        st.header("Key Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if selected_bench['milvus_load_time'] and selected_bench['weaviate_load_time']:
                speedup = selected_bench['weaviate_load_time'] / selected_bench['milvus_load_time']
                st.metric(
                    "Load Time Winner",
                    "🏆 Milvus" if speedup > 1 else "🏆 Weaviate",
                    f"{speedup:.1f}x faster" if speedup > 1 else f"{1/speedup:.1f}x faster"
                )
        
        with col2:
            if comparison_df is not None:
                milvus_p50 = comparison_df[(comparison_df['Database'] == 'Milvus') & 
                                            (comparison_df['Test'] == 'k10_nofilter')]['P50 (ms)'].values
                weaviate_p50 = comparison_df[(comparison_df['Database'] == 'Weaviate') & 
                                              (comparison_df['Test'] == 'k10_nofilter')]['P50 (ms)'].values
                
                if len(milvus_p50) > 0 and len(weaviate_p50) > 0:
                    speedup = weaviate_p50[0] / milvus_p50[0]
                    st.metric(
                        "Query Latency Winner",
                        "🏆 Milvus" if speedup > 1 else "🏆 Weaviate",
                        f"{speedup:.1f}x faster" if speedup > 1 else f"{1/speedup:.1f}x faster"
                    )
        
        with col3:
            if comparison_df is not None:
                milvus_qps = comparison_df[(comparison_df['Database'] == 'Milvus') & 
                                            (comparison_df['Test'] == 'k10_nofilter')]['QPS'].values
                weaviate_qps = comparison_df[(comparison_df['Database'] == 'Weaviate') & 
                                              (comparison_df['Test'] == 'k10_nofilter')]['QPS'].values
                
                if len(milvus_qps) > 0 and len(weaviate_qps) > 0:
                    speedup = milvus_qps[0] / weaviate_qps[0]
                    st.metric(
                        "Throughput Winner",
                        "🏆 Milvus" if speedup > 1 else "🏆 Weaviate",
                        f"{speedup:.1f}x higher QPS" if speedup > 1 else f"{1/speedup:.1f}x higher QPS"
                    )
        
        with col4:
            if 10 in selected_bench['recall']:
                m_recall = selected_bench['recall'][10]['Milvus']
                w_recall = selected_bench['recall'][10]['Weaviate']
                winner = "Weaviate" if w_recall > m_recall else "🏆 Milvus"
                diff = abs(w_recall - m_recall) * 100
                st.metric(
                    "Recall@10 Winner",
                    winner,
                    f"+{diff:.1f}% accuracy"
                )
        
        st.divider()
        
        # Loading Performance Section
        if show_loading:
            st.header("Data Loading Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Load Time")
                load_df = pd.DataFrame({
                    'Database': ['Milvus', 'Weaviate'],
                    'Load Time (s)': [selected_bench['milvus_load_time'], selected_bench['weaviate_load_time']]
                })
                fig = px.bar(
                    load_df,
                    x='Database',
                    y='Load Time (s)',
                    color='Database',
                    color_discrete_map={'Milvus': '#00b894', 'Weaviate': '#e17055'}
                )
                fig.update_layout(showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Peak Memory Usage")
                mem_df = pd.DataFrame({
                    'Database': ['Milvus', 'Weaviate'],
                    'Peak Memory (MB)': [selected_bench['milvus_peak_memory'], selected_bench['weaviate_peak_memory']]
                })
                fig = px.bar(
                    mem_df,
                    x='Database',
                    y='Peak Memory (MB)',
                    color='Database',
                    color_discrete_map={'Milvus': '#00b894', 'Weaviate': '#e17055'}
                )
                fig.update_layout(showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
        
        # Query Performance Section
        if show_latency and comparison_df is not None:
            st.header("Query Latency Performance")
            
            latency_metric = st.radio(
                "Select Latency Percentile",
                ['P50 (ms)', 'P95 (ms)', 'P99 (ms)', 'Mean (ms)'],
                horizontal=True
            )
            
            test_filter = st.multiselect(
                "Filter Tests",
                options=comparison_df['Test'].unique().tolist(),
                default=comparison_df['Test'].unique().tolist()
            )
            
            filtered_df = comparison_df[comparison_df['Test'].isin(test_filter)]
            
            fig = create_latency_chart(filtered_df, latency_metric)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
        
        # QPS Section
        if show_qps and comparison_df is not None:
            st.header("Queries Per Second (Throughput)")
            
            fig = create_qps_chart(comparison_df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
        
        # Recall Section
        if show_recall:
            st.header("Recall@K Accuracy")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Try to use recall from benchmark data first
                if selected_bench['recall']:
                    fig = create_recall_chart(selected_bench['recall'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                elif recall_df is not None:
                    fig = create_recall_chart_from_df(recall_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No recall data available for this run.")
            
            with col2:
                st.subheader("Recall Values")
                if selected_bench['recall']:
                    recall_rows = []
                    for k in sorted(selected_bench['recall'].keys()):
                        recall_rows.append({
                            'K': k,
                            'Milvus': selected_bench['recall'][k]['Milvus'],
                            'Weaviate': selected_bench['recall'][k]['Weaviate']
                        })
                    recall_table = pd.DataFrame(recall_rows)
                    styled_df = recall_table.style.format({
                        'Milvus': '{:.4f}',
                        'Weaviate': '{:.4f}'
                    }).background_gradient(
                        subset=['Milvus', 'Weaviate'],
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=1
                    )
                    st.dataframe(styled_df, use_container_width=True)
                elif recall_df is not None:
                    styled_df = recall_df.style.format({
                        'Milvus': '{:.4f}',
                        'Weaviate': '{:.4f}'
                    }).background_gradient(
                        subset=['Milvus', 'Weaviate'],
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=1
                    )
                    st.dataframe(styled_df, use_container_width=True)
            
            st.divider()
        
        # Raw Data Section
        if show_raw_data:
            st.header("Raw Data")
            
            if comparison_df is not None:
                st.subheader("Query Performance Data")
                st.dataframe(comparison_df, use_container_width=True)
            
            if recall_df is not None:
                st.subheader("Recall Data")
                st.dataframe(recall_df, use_container_width=True)
    
    # Recommendations Section
    st.header("💡 AI-Powered Recommendations")
    
    # Get API key from environment or sidebar input
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    
    with st.sidebar:
        st.divider()
        st.subheader("🤖 AI Analysis")
        api_key = st.text_input(
            "OpenAI API Key",
            value=env_api_key,
            type="password",
            placeholder="sk-...",
            help="API key loaded from .env file. You can also enter a different key here."
        )
        analyze_button = st.button("🔍 Analyze with AI", use_container_width=True)
    
    if api_key and analyze_button:
        with st.spinner("🤖 AI is analyzing your benchmark data..."):
            ai_recommendations = get_ai_recommendations(comparison_df, recall_df, api_key)
            if ai_recommendations:
                st.session_state['ai_recommendations'] = ai_recommendations
    
    # Display AI recommendations if available
    if 'ai_recommendations' in st.session_state and st.session_state['ai_recommendations']:
        st.markdown(st.session_state['ai_recommendations'])
    else:
        # Default recommendations when no AI analysis
        st.info("💡 Enter your OpenAI API key in the sidebar and click 'Analyze with AI' to get personalized recommendations based on your benchmark data.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Choose **Milvus** if:
            - Maximum query performance is critical
            - You need fine-grained index control
            - GPU acceleration is needed
            - High throughput is a priority
            """)
        
        with col2:
            st.markdown("""
            ### Choose **Weaviate** if:
            - Search accuracy is paramount
            - You need GraphQL API support
            - Hybrid search (vector + keyword) is important
            - Easier setup and management is preferred
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built by minageus, cobra, mountzouris</p>
        <p>Run <code>python run_benchmark.py</code> to generate new benchmarks</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
