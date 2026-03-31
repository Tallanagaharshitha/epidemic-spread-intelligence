import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    """Create visualizations for epidemic data"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set1
    
    def plot_predictions(self, historical_data, predictions):
        """Plot historical data with predictions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Epidemic Curve', 'Daily New Cases', 
                          'Regional Comparison', 'Growth Rate'),
            specs=[[{'secondary_y': True}, {}], [{}, {}]]
        )
        
        # Epidemic Curve
        for i, region in enumerate(historical_data['region'].unique()):
            region_hist = historical_data[historical_data['region'] == region]
            region_pred = predictions[predictions['region'] == region]
            
            # Historical
            fig.add_trace(
                go.Scatter(
                    x=region_hist['date'],
                    y=region_hist['confirmed'],
                    name=f"{region} (Historical)",
                    line=dict(color=self.colors[i % len(self.colors)], dash='solid'),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=region_pred['date'],
                    y=region_pred['predicted_cases'],
                    name=f"{region} (Predicted)",
                    line=dict(color=self.colors[i % len(self.colors)], dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Daily New Cases
        for i, region in enumerate(historical_data['region'].unique()):
            region_hist = historical_data[historical_data['region'] == region]
            
            fig.add_trace(
                go.Bar(
                    x=region_hist['date'].tail(30),
                    y=region_hist['new_cases'].tail(30),
                    name=f"{region} New Cases",
                    marker_color=self.colors[i % len(self.colors)],
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # Regional Comparison (latest data)
        latest_data = historical_data[historical_data['date'] == historical_data['date'].max()]
        
        fig.add_trace(
            go.Bar(
                x=latest_data['region'],
                y=latest_data['confirmed'],
                name='Total Cases',
                marker_color=self.colors,
                text=latest_data['confirmed'],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Growth Rate
        for i, region in enumerate(historical_data['region'].unique()):
            region_hist = historical_data[historical_data['region'] == region]
            
            fig.add_trace(
                go.Scatter(
                    x=region_hist['date'].tail(30),
                    y=region_hist['growth_rate'].tail(30),
                    name=f"{region} Growth Rate",
                    line=dict(color=self.colors[i % len(self.colors)]),
                    mode='lines+markers'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Epidemic Analysis Dashboard",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Region", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        
        fig.update_yaxes(title_text="Confirmed Cases", row=1, col=1)
        fig.update_yaxes(title_text="New Cases", row=1, col=2)
        fig.update_yaxes(title_text="Total Cases", row=2, col=1)
        fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=2)
        
        return fig
    
    def plot_r0_chart(self, r0_data):
        """Plot reproduction number by region"""
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=r0_data['region'],
            y=r0_data['R0'],
            marker_color=['red' if r > 1 else 'green' for r in r0_data['R0']],
            text=r0_data['R0'].round(2),
            textposition='auto',
            name='R0'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=1, 
            line_dash="dash", 
            line_color="black",
            annotation_text="R0 = 1 (Threshold)",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title="Reproduction Number (R0) by Region",
            xaxis_title="Region",
            yaxis_title="R0 Value",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_growth_rates(self, data):
        """Plot growth rate trends"""
        fig = go.Figure()
        
        for region in data['region'].unique():
            region_data = data[data['region'] == region]
            
            fig.add_trace(go.Scatter(
                x=region_data['date'].tail(60),
                y=region_data['growth_rate'].tail(60),
                mode='lines',
                name=region,
                line=dict(width=2)
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Daily Growth Rate Trends (Last 60 Days)",
            xaxis_title="Date",
            yaxis_title="Growth Rate (%)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_intervention_impact(self, baseline, scenarios):
        """Plot intervention impact comparison"""
        fig = go.Figure()
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=baseline['date'],
            y=baseline['cases'],
            mode='lines',
            name='Baseline (No Interventions)',
            line=dict(color='red', width=3, dash='solid')
        ))
        
        # Intervention scenarios
        colors = ['green', 'blue', 'orange', 'purple']
        for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
            fig.add_trace(go.Scatter(
                x=scenario_data['date'],
                y=scenario_data['cases'],
                mode='lines',
                name=scenario_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Intervention Impact Simulation",
            xaxis_title="Date",
            yaxis_title="Predicted Cases",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_heatmap(self, data):
        """Create heatmap of spread by region and time"""
        # Pivot data for heatmap
        pivot_data = data.pivot_table(
            values='incidence_per_100k',
            index='region',
            columns='date',
            aggfunc='mean'
        ).fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn_r',
            zmid=50,
            colorbar=dict(title="Incidence per 100k")
        ))
        
        fig.update_layout(
            title="Spatial-Temporal Spread Heatmap",
            xaxis_title="Date",
            yaxis_title="Region",
            height=400
        )
        
        return fig
    
    def plot_age_distribution(self, age_data):
        """Plot age distribution of cases"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cases by Age Group', 'CFR by Age Group'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Cases by age
        fig.add_trace(
            go.Bar(
                x=age_data['age_group'],
                y=age_data['cases'],
                name='Cases',
                marker_color='skyblue',
                text=age_data['cases'],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # CFR by age
        fig.add_trace(
            go.Bar(
                x=age_data['age_group'],
                y=age_data['cfr'],
                name='CFR (%)',
                marker_color='lightcoral',
                text=age_data['cfr'].round(1),
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Age Distribution Analysis",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Age Group", row=1, col=1)
        fig.update_xaxes(title_text="Age Group", row=1, col=2)
        fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
        fig.update_yaxes(title_text="Case Fatality Rate (%)", row=1, col=2)
        
        return fig