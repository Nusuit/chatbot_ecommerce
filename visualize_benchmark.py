import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="RAG Benchmark Visualization",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 16px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Load benchmark results
try:
    df = pd.read_csv('benchmark_results.csv')
    st.title("📊 RAG Benchmark Results")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}</div>
            <div class="metric-label">Average Relevance Score</div>
        </div>
        """.format(df['relevance_score'].mean()), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}</div>
            <div class="metric-label">Average Response Time (s)</div>
        </div>
        """.format(df['response_time'].mean()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}</div>
            <div class="metric-label">Average Retrieval Time (s)</div>
        </div>
        """.format(df['retrieval_time'].mean()), unsafe_allow_html=True)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Relevance Score Distribution",
            "Response Time vs Relevance",
            "Query Length vs Response Time",
            "Retrieval Time Distribution"
        )
    )
    
    # 1. Relevance Score Distribution
    fig.add_trace(
        go.Histogram(x=df['relevance_score'], name="Relevance Scores",
                    marker_color='#1f77b4'),
        row=1, col=1
    )
    
    # 2. Response Time vs Relevance
    fig.add_trace(
        go.Scatter(x=df['response_time'], y=df['relevance_score'],
                  mode='markers', name="Time vs Relevance",
                  marker=dict(color='#2ca02c')),
        row=1, col=2
    )
    
    # 3. Query Length vs Response Time
    df['query_length'] = df['query'].str.len()
    fig.add_trace(
        go.Scatter(x=df['query_length'], y=df['response_time'],
                  mode='markers', name="Query Length vs Time",
                  marker=dict(color='#ff7f0e')),
        row=2, col=1
    )
    
    # 4. Retrieval Time Distribution
    fig.add_trace(
        go.Histogram(x=df['retrieval_time'], name="Retrieval Times",
                    marker_color='#9467bd'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df)

except FileNotFoundError:
    st.error("⚠️ benchmark_results.csv not found. Please run the benchmark first.")
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}") 