import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="FAISS Index Benchmark Results",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 FAISS Index Benchmark Results")
st.markdown("""
This dashboard shows the comparison between different FAISS index types:
- **FLAT**: Basic exhaustive search index
- **IVF**: Inverted File index with clustering
- **HNSW**: Hierarchical Navigable Small World graph
""")

# Create benchmark data
data = {
    'Index Type': ['FLAT', 'IVF', 'HNSW'],
    'Precision': [60.00, 52.00, 20.00],
    'Recall': [100.00, 86.67, 33.33],
    'Search Time (ms)': [37.07, 10.33, 0.80],
    'Average Distance': [80.3820, 80.5091, 83.2929]
}
df = pd.DataFrame(data)

# Create two columns
col1, col2 = st.columns(2)

# Precision & Recall Bar Chart
with col1:
    st.subheader("Precision & Recall Comparison")
    fig_pr = go.Figure(data=[
        go.Bar(name='Precision', x=df['Index Type'], y=df['Precision']),
        go.Bar(name='Recall', x=df['Index Type'], y=df['Recall'])
    ])
    fig_pr.update_layout(barmode='group', yaxis_title='Percentage (%)')
    st.plotly_chart(fig_pr, use_container_width=True)

# Search Time Comparison
with col2:
    st.subheader("Search Time Comparison")
    fig_time = px.bar(df, x='Index Type', y='Search Time (ms)',
                     color='Index Type',
                     labels={'Search Time (ms)': 'Time (milliseconds)'})
    st.plotly_chart(fig_time, use_container_width=True)

# Average Distance Comparison
st.subheader("Average Distance Comparison")
fig_dist = px.line(df, x='Index Type', y='Average Distance', markers=True)
fig_dist.update_traces(line_width=3)
st.plotly_chart(fig_dist, use_container_width=True)

# Detailed Results Table
st.subheader("Detailed Results")
st.dataframe(
    df.style.format({
        'Precision': '{:.2f}%',
        'Recall': '{:.2f}%',
        'Search Time (ms)': '{:.2f}',
        'Average Distance': '{:.4f}'
    }),
    use_container_width=True
)

# Key Findings
st.subheader("🔍 Key Findings")
st.markdown("""
- **FLAT Index** provides the highest accuracy with 60% precision and 100% recall, but slowest search time (37.07ms)
- **IVF Index** offers a good balance with 52% precision, 86.67% recall, and improved search time (10.33ms)
- **HNSW Index** is the fastest (0.80ms) but has lower accuracy metrics
""")

# Recommendations
st.subheader("💡 Recommendations")
st.info("""
Based on the benchmark results, the **IVF Index** appears to be the best choice for most use cases as it provides:
1. Reasonable accuracy (only slightly lower than FLAT)
2. Significantly improved search time (3.6x faster than FLAT)
3. Similar average distance scores to FLAT index
""") 