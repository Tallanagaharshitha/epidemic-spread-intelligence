import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from data_processor import DataProcessor
from opencv_extractor import CurveExtractor
from epidemic_models import EpidemicModels
from visualization import Visualizer
from intervention_simulator import InterventionSimulator
from report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Epidemic Intelligence Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-icon {
        font-size: 2rem;
        float: right;
    }
    
    .info-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #1f1f1f;
        margin-right: 2rem;
        border-left: 5px solid #667eea;
    }
    
    .chat-icon {
        font-size: 1.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'intervention_results' not in st.session_state:
    st.session_state.intervention_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Sidebar
with st.sidebar:
    st.markdown("### 🦠 Epidemic Intelligence")
    st.markdown("---")
    
    st.markdown("### 📊 Data Source")
    input_method = st.radio(
        "Select input method:",
        ["📁 Upload CSV/Excel", "🖼️ Upload Image (OpenCV)", "📊 Sample Data"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Model Settings")
    prediction_days = st.slider("Prediction Horizon (Days)", 7, 90, 30, key="pred_days")
    model_type = st.selectbox(
        "Epidemic Model",
        ["SIR Model", "SEIR Model", "ARIMA"],
        key="model_type"
    )
    
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# Main content
st.markdown('<div class="main-header">🦠 Epidemic Spread Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Epidemic Analysis & Interactive Recommendations")
st.markdown("---")

# Data loading section
if not st.session_state.data_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Welcome to Epidemic Intelligence")
        st.markdown("""
        This advanced dashboard provides:
        - 🔮 **AI-powered predictions** using multiple epidemic models
        - 💉 **Intervention simulation** to test policy impacts  
        - 💬 **Interactive AI Chat** for real-time insights
        - 📊 **Interactive visualizations** for data exploration
        - 📑 **Comprehensive reports** in multiple formats
        """)
        st.markdown("---")
        st.markdown("#### Quick Start:")
        st.markdown("1. Upload your epidemic data or use sample data")
        st.markdown("2. Configure model parameters in the sidebar")
        st.markdown("3. Click 'Run Analysis' to generate insights")
        st.markdown('</div>', unsafe_allow_html=True)

# Data processing based on input method
if input_method == "📁 Upload CSV/Excel":
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file:
        processor = DataProcessor()
        if uploaded_file.name.endswith('.csv'):
            df = processor.load_csv(uploaded_file)
        else:
            df = processor.load_excel(uploaded_file)
        
        if df is not None:
            st.session_state.data = df
            st.session_state.processor = processor
            st.session_state.data_loaded = True
            st.session_state.analysis_done = False
            st.success("✅ Data loaded successfully!")
            
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

elif input_method == "🖼️ Upload Image (OpenCV)":
    uploaded_image = st.file_uploader(
        "Upload epidemic curve image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Extract Data from Image", use_container_width=True):
                extractor = CurveExtractor()
                image = extractor.load_image(uploaded_image)
                
                if image is not None:
                    with st.spinner("Extracting curve data..."):
                        extracted_data = extractor.extract_curve_data(image)
                        
                        if extracted_data is not None:
                            st.session_state.data = extracted_data
                            st.session_state.processor = DataProcessor()
                            st.session_state.data = st.session_state.processor.validate_and_clean(extracted_data)
                            st.session_state.data_loaded = True
                            st.session_state.analysis_done = False
                            st.success("✅ Data extracted successfully!")

else:  # Sample data
    st.info("📊 Using sample epidemic data for demonstration")
    
    # Generate realistic sample data
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    regions = ['North Region', 'South Region', 'East Region', 'West Region', 'Central Region']
    
    data = []
    np.random.seed(42)
    
    for region in regions:
        base_cases = np.random.randint(50, 200)
        growth_factor = np.random.uniform(1.02, 1.08)
        
        for i, date in enumerate(dates):
            growth = np.exp(i/45) * np.random.normal(1, 0.05)
            confirmed = int(base_cases * growth * (growth_factor ** i))
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * i / 30)
            confirmed = int(confirmed * seasonal)
            recovered = int(confirmed * np.random.uniform(0.4, 0.7))
            deaths = int(confirmed * np.random.uniform(0.01, 0.04))
            
            data.append({
                'date': date,
                'region': region,
                'confirmed': max(0, confirmed),
                'recovered': max(0, recovered),
                'deaths': max(0, deaths),
                'population': np.random.randint(500000, 2000000)
            })
    
    df = pd.DataFrame(data)
    processor = DataProcessor()
    df = processor.validate_and_clean(df)
    st.session_state.data = df
    st.session_state.processor = processor
    st.session_state.data_loaded = True
    st.session_state.analysis_done = False
    
    with st.expander("📊 Sample Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

# Display metrics if data loaded
if st.session_state.data_loaded and st.session_state.data is not None:
    try:
        stats = st.session_state.processor.get_summary_stats()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-label">Total Cases</div>
                <div class="metric-value">{stats.get('total_confirmed', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔄</div>
                <div class="metric-label">Active Cases</div>
                <div class="metric-value">{stats.get('active_cases', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-label">Recovered</div>
                <div class="metric-value">{stats.get('total_recovered', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💀</div>
                <div class="metric-label">Deaths</div>
                <div class="metric-value">{stats.get('total_deaths', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-label">CFR</div>
                <div class="metric-value">{stats.get('cfr', 0):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    except Exception as e:
        st.warning(f"Could not display metrics: {str(e)}")

# Run analysis
if run_analysis and st.session_state.data_loaded and st.session_state.data is not None:
    with st.spinner("Running epidemic models and generating predictions..."):
        try:
            models = EpidemicModels(st.session_state.data)
            
            all_predictions = []
            regions_list = list(st.session_state.data['region'].unique())
            
            progress_bar = st.progress(0)
            for idx, region in enumerate(regions_list):
                region_data = st.session_state.data[st.session_state.data['region'] == region]
                
                if model_type == "SIR Model":
                    pred = models.sir_model(region_data, days=prediction_days)
                elif model_type == "SEIR Model":
                    pred = models.seir_model(region_data, days=prediction_days)
                else:
                    pred = models.arima_forecast(region_data, days=prediction_days)
                
                pred['region'] = region
                all_predictions.append(pred)
                progress_bar.progress((idx + 1) / len(regions_list))
            
            progress_bar.empty()
            st.session_state.predictions = pd.concat(all_predictions, ignore_index=True)
            st.session_state.models = models
            st.session_state.analysis_done = True
            
            st.success("✅ Analysis complete! Results displayed below.")
            st.balloons()
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Display results if analysis has been done
if st.session_state.analysis_done and st.session_state.predictions is not None:
    st.markdown("## 🔬 Epidemic Analysis Results")
    
    # Get regions list
    regions_list = list(st.session_state.data['region'].unique())
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predictions",
        "🦠 Model Analysis",
        "💉 Intervention",
        "💬 AI Chat",
        "📑 Reports"
    ])
    
    # Tab 1: Predictions & Visualizations
    with tab1:
        st.markdown("### 📈 Epidemic Spread Predictions")
        
        # Define colors
        colors_rgb = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44),
            (214, 39, 40), (148, 103, 189), (140, 86, 75),
            (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
        ]
        
        fig1 = go.Figure()
        
        for i, region in enumerate(regions_list):
            hist_data = st.session_state.data[st.session_state.data['region'] == region]
            pred_data = st.session_state.predictions[st.session_state.predictions['region'] == region]
            
            r, g, b = colors_rgb[i % len(colors_rgb)]
            
            fig1.add_trace(go.Scatter(
                x=hist_data['date'],
                y=hist_data['confirmed'],
                mode='lines',
                name=f'{region} (Actual)',
                line=dict(color=f'rgb({r}, {g}, {b})', width=3),
                fill='tozeroy',
                fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
                legendgroup=region
            ))
            
            fig1.add_trace(go.Scatter(
                x=pred_data['date'],
                y=pred_data['predicted_cases'],
                mode='lines',
                name=f'{region} (Forecast)',
                line=dict(color=f'rgb({r}, {g}, {b})', width=2, dash='dash'),
                legendgroup=region,
                showlegend=False
            ))
        
        fig1.update_layout(
            title="Epidemic Trajectory with Forecast",
            xaxis_title="Date",
            yaxis_title="Confirmed Cases",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Daily New Cases")
            fig_new = go.Figure()
            
            for i, region in enumerate(regions_list):
                region_data = st.session_state.data[st.session_state.data['region'] == region]
                r, g, b = colors_rgb[i % len(colors_rgb)]
                fig_new.add_trace(go.Bar(
                    x=region_data['date'].tail(60),
                    y=region_data['new_cases'].tail(60),
                    name=region,
                    opacity=0.7,
                    marker=dict(color=f'rgb({r}, {g}, {b})')
                ))
            
            fig_new.update_layout(
                title="New Cases Distribution",
                xaxis_title="Date",
                yaxis_title="New Cases",
                height=400,
                template='plotly_white',
                barmode='group'
            )
            st.plotly_chart(fig_new, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Growth Rate Trends")
            fig_growth = go.Figure()
            
            for i, region in enumerate(regions_list):
                region_data = st.session_state.data[st.session_state.data['region'] == region]
                r, g, b = colors_rgb[i % len(colors_rgb)]
                fig_growth.add_trace(go.Scatter(
                    x=region_data['date'].tail(60),
                    y=region_data['growth_rate'].tail(60),
                    mode='lines+markers',
                    name=region,
                    line=dict(color=f'rgb({r}, {g}, {b})', width=2),
                    marker=dict(size=4)
                ))
            
            fig_growth.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig_growth.update_layout(
                title="Growth Rate Trends",
                xaxis_title="Date",
                yaxis_title="Growth Rate (%)",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with st.expander("📋 Detailed Predictions Table"):
            st.dataframe(st.session_state.predictions, use_container_width=True)
    
    # Tab 2: Model Analysis
    with tab2:
        st.markdown("### 🧬 Advanced Model Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Reproduction Number (R₀)")
            
            for region in regions_list:
                region_data = st.session_state.data[st.session_state.data['region'] == region]
                r0 = st.session_state.models.calculate_r0(region_data)
                
                if r0 > 1.5:
                    color = "red"
                    status = "⚠️ High Risk"
                elif r0 > 1:
                    color = "orange"
                    status = "📈 Moderate Risk"
                else:
                    color = "green"
                    status = "✅ Controlled"
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=r0,
                    title={'text': f"{region}"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 3]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgreen"},
                            {'range': [1, 1.5], 'color': "orange"},
                            {'range': [1.5, 3], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1
                        }
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=50, r=50, t=50, b=50))
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption(f"Status: {status}")
        
        with col2:
            st.markdown("#### Peak Predictions")
            peak_data = []
            for region in st.session_state.predictions['region'].unique():
                region_pred = st.session_state.predictions[st.session_state.predictions['region'] == region]
                peak = region_pred.loc[region_pred['predicted_cases'].idxmax()]
                peak_data.append({
                    'Region': region,
                    'Peak Date': peak['date'].strftime('%Y-%m-%d'),
                    'Peak Cases': int(peak['predicted_cases']),
                    'Days to Peak': (peak['date'] - pd.Timestamp.now()).days
                })
            
            peak_df = pd.DataFrame(peak_data).sort_values('Days to Peak')
            
            fig_peak = go.Figure()
            fig_peak.add_trace(go.Bar(
                x=peak_df['Region'],
                y=peak_df['Peak Cases'],
                text=peak_df['Peak Cases'],
                textposition='auto',
                marker_color=['#ff6b6b' if d < 30 else '#4ecdc4' for d in peak_df['Days to Peak']]
            ))
            fig_peak.update_layout(
                title="Projected Peak Cases by Region",
                xaxis_title="Region",
                yaxis_title="Peak Cases",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_peak, use_container_width=True)
            st.dataframe(peak_df, use_container_width=True)
        
        st.markdown("### 🌍 Spatial-Temporal Heatmap")
        
        heatmap_data = st.session_state.data.pivot_table(
            values='incidence_per_100k',
            index='region',
            columns=st.session_state.data['date'].dt.strftime('%Y-%m-%d'),
            aggfunc='mean'
        ).fillna(0)
        
        if len(heatmap_data.columns) > 30:
            heatmap_data = heatmap_data.iloc[:, -30:]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn_r',
            zmid=50,
            colorbar=dict(title="Incidence per 100k"),
            hoverongaps=False
        ))
        
        fig_heatmap.update_layout(
            title="Epidemic Intensity Over Time",
            xaxis_title="Date",
            yaxis_title="Region",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Tab 3: Intervention Simulation
    with tab3:
        st.markdown("### 💉 Interactive Intervention Simulator")
        
        simulator = InterventionSimulator(st.session_state.data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            interventions = st.multiselect(
                "Select interventions:",
                ["Vaccination Campaign", "Lockdown Measures", "Social Distancing", 
                 "Mask Mandates", "Travel Restrictions", "Testing & Tracing"],
                default=["Vaccination Campaign", "Social Distancing"],
                key="interventions_select"
            )
        
        with col2:
            strength = st.slider("Effectiveness (%)", 0, 100, 50, key="intervention_strength")
        
        if st.button("🔄 Run Simulation", use_container_width=True, key="run_simulation"):
            if interventions:
                with st.spinner("Running simulation..."):
                    results = simulator.simulate_interventions(interventions, strength / 100)
                    st.session_state.intervention_results = results
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Cases Averted", f"{int(results['cases_averted']):,}", f"{results['reduction_percentage']:.1f}%")
                with col2:
                    st.metric("Peak Reduction", f"{results['peak_reduction']:.1f}%")
                with col3:
                    st.metric("Peak Delay", f"{results['peak_delay']} days")
                with col4:
                    st.metric("Cases with Intervention", f"{int(results['intervention_cases']):,}")
                
                fig = simulator.plot_intervention_comparison()
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please select at least one intervention")
    
    # Tab 4: AI Chat Assistant
    with tab4:
        st.markdown("### 💬 AI Epidemic Intelligence Assistant")
        
        # Simple AI response function
        def get_ai_response(question, data, stats):
            question_lower = question.lower()
            total_cases = stats.get('total_confirmed', 0)
            active_cases = stats.get('active_cases', 0)
            
            if 'situation' in question_lower or 'overview' in question_lower:
                return f"📊 Current Situation: {total_cases:,} total cases, {active_cases:,} active cases. The epidemic is currently active across {stats.get('regions_affected', 0)} regions."
            elif 'intervention' in question_lower or 'recommend' in question_lower:
                return f"💉 Recommended: Increase testing, implement social distancing, and prepare healthcare surge capacity. These measures could reduce cases by 30-50%."
            elif 'peak' in question_lower:
                return f"📅 Based on current trends, the peak is expected in approximately 2-4 weeks. Continue monitoring closely."
            else:
                return f"🤔 I can help with: current situation, interventions, peak predictions, and regional analysis. What would you like to know?"
        
        if not st.session_state.chat_history:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": "👋 Hello! Ask me about the current situation, interventions, or peak predictions."
            })
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="chat-icon">👤</div>
                    <div><strong>You:</strong><br>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="chat-icon">🤖</div>
                    <div><strong>AI Assistant:</strong><br>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        user_question = st.text_input("Ask me anything:", key="chat_input")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ask_button = st.button("Send", use_container_width=True, key="send_message")
        
        if ask_button and user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            stats = st.session_state.processor.get_summary_stats()
            response = get_ai_response(user_question, st.session_state.data, stats)
            st.session_state.chat_history.append({"role": "bot", "content": response})
            st.rerun()
    
    # Tab 5: Reports
    with tab5:
        st.markdown("### 📑 Generate Reports")
        
        report_format = st.selectbox("Format:", ["PDF Report", "Excel Data Export", "HTML Dashboard"], key="report_format")
        custom_name = st.text_input("Report Name", value=f"Epidemic_Report_{datetime.datetime.now().strftime('%Y%m%d')}", key="report_name")
        
        if st.button("📥 Generate Report", use_container_width=True, key="generate_report"):
            with st.spinner("Generating..."):
                generator = ReportGenerator()
                
                if "PDF" in report_format:
                    report_data = generator.generate_pdf_report(
                        data=st.session_state.data,
                        predictions=st.session_state.predictions,
                        stats=st.session_state.processor.get_summary_stats(),
                        recommendations=None
                    )
                    st.download_button("Download PDF", report_data, f"{custom_name}.pdf", "application/pdf", key="download_pdf")
                elif "Excel" in report_format:
                    excel_data = generator.generate_excel_report(
                        data=st.session_state.data,
                        predictions=st.session_state.predictions
                    )
                    st.download_button("Download Excel", excel_data, f"{custom_name}.xlsx", 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel")
                else:
                    html_report = generator.generate_html_report(
                        data=st.session_state.data,
                        predictions=st.session_state.predictions,
                        stats=st.session_state.processor.get_summary_stats()
                    )
                    st.download_button("Download HTML", html_report, f"{custom_name}.html", "text/html", key="download_html")
        
        st.markdown("### 💾 Quick Export")
        csv = st.session_state.data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Raw Data (CSV)", csv, "epidemic_data.csv", "text/csv", key="download_csv")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🦠 Epidemic Spread Intelligence System v2.0 | Powered by Advanced Analytics & AI</p>
    <p>Click 'Run Analysis' to generate predictions and insights</p>
</div>
""", unsafe_allow_html=True)






