import numpy as np
import pandas as pd
import streamlit as st

class InterventionSimulator:
    """Simulate the impact of various interventions"""
    
    def __init__(self, data):
        self.data = data
        self.baseline_cases = None
        self.intervention_cases = None
        self.intervention_effects = {
            'Vaccination': 0.7,  # 70% reduction in transmission
            'Lockdown': 0.8,      # 80% reduction in contacts
            'Social Distancing': 0.5,  # 50% reduction
            'Mask Mandate': 0.4,   # 40% reduction
            'Travel Restrictions': 0.3,  # 30% reduction
            'School Closure': 0.25,  # 25% reduction
            'Testing & Tracing': 0.35,  # 35% reduction
            'Quarantine': 0.45  # 45% reduction
        }
        
    def simulate_interventions(self, selected_interventions, strength=0.5):
        """Simulate the combined effect of selected interventions"""
        
        # Calculate combined effect
        combined_effect = 1.0
        for intervention in selected_interventions:
            if intervention in self.intervention_effects:
                # Interventions have diminishing returns when combined
                effect = self.intervention_effects[intervention] * strength
                combined_effect *= (1 - effect)
        
        combined_effect = 1 - combined_effect  # Convert to reduction percentage
        
        # Apply to baseline projections
        self._generate_baseline()
        self._apply_interventions(combined_effect)
        
        # Calculate impact metrics
        results = self._calculate_impact()
        
        return results
    
    def _generate_baseline(self):
        """Generate baseline projection without interventions"""
        # Simple exponential growth model for baseline
        latest_cases = self.data.groupby('date')['confirmed'].sum().iloc[-1]
        growth_rate = 0.05  # 5% daily growth
        
        dates = pd.date_range(
            start=self.data['date'].max() + pd.Timedelta(days=1),
            periods=90,
            freq='D'
        )
        
        baseline = []
        cases = latest_cases
        for date in dates:
            cases = cases * (1 + growth_rate)
            baseline.append({
                'date': date,
                'cases': cases
            })
        
        self.baseline_cases = pd.DataFrame(baseline)
    
    def _apply_interventions(self, effect):
        """Apply intervention effect to baseline"""
        # Intervention effect grows over time
        intervention_cases = []
        
        for i, row in self.baseline_cases.iterrows():
            # Effect ramps up over first 30 days
            ramp_up = min(1.0, i / 30)
            current_effect = effect * ramp_up
            
            # Apply effect
            reduced_cases = row['cases'] * (1 - current_effect)
            intervention_cases.append({
                'date': row['date'],
                'cases': reduced_cases
            })
        
        self.intervention_cases = pd.DataFrame(intervention_cases)
    
    def _calculate_impact(self):
        """Calculate impact metrics"""
        total_baseline = self.baseline_cases['cases'].sum()
        total_intervention = self.intervention_cases['cases'].sum()
        
        cases_averted = total_baseline - total_intervention
        
        # Peak reduction
        baseline_peak = self.baseline_cases['cases'].max()
        intervention_peak = self.intervention_cases['cases'].max()
        peak_reduction = (baseline_peak - intervention_peak) / baseline_peak * 100
        
        # Delay in peak
        baseline_peak_date = self.baseline_cases.loc[
            self.baseline_cases['cases'].idxmax(), 'date'
        ]
        intervention_peak_date = self.intervention_cases.loc[
            self.intervention_cases['cases'].idxmax(), 'date'
        ]
        peak_delay = (intervention_peak_date - baseline_peak_date).days
        
        results = {
            'baseline_cases': total_baseline,
            'intervention_cases': total_intervention,
            'cases_averted': cases_averted,
            'reduction_percentage': (cases_averted / total_baseline * 100),
            'peak_reduction': peak_reduction,
            'peak_delay': peak_delay,
            'baseline_peak_date': baseline_peak_date,
            'intervention_peak_date': intervention_peak_date
        }
        
        return results
    
    def plot_intervention_comparison(self):
        """Create visualization of intervention impact"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=self.baseline_cases['date'],
            y=self.baseline_cases['cases'],
            mode='lines',
            name='Without Interventions',
            line=dict(color='red', width=3)
        ))
        
        # With interventions
        fig.add_trace(go.Scatter(
            x=self.intervention_cases['date'],
            y=self.intervention_cases['cases'],
            mode='lines',
            name='With Interventions',
            line=dict(color='green', width=3)
        ))
        
        # Add peak markers
        fig.add_trace(go.Scatter(
            x=[self.baseline_cases.loc[self.baseline_cases['cases'].idxmax(), 'date']],
            y=[self.baseline_cases['cases'].max()],
            mode='markers',
            name='Baseline Peak',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.add_trace(go.Scatter(
            x=[self.intervention_cases.loc[self.intervention_cases['cases'].idxmax(), 'date']],
            y=[self.intervention_cases['cases'].max()],
            mode='markers',
            name='Intervention Peak',
            marker=dict(color='green', size=12, symbol='star')
        ))
        
        # Fill between curves
        fig.add_trace(go.Scatter(
            x=pd.concat([self.baseline_cases['date'], self.intervention_cases['date'][::-1]]),
            y=pd.concat([self.baseline_cases['cases'], self.intervention_cases['cases'][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Cases Averted',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Intervention Impact Analysis",
            xaxis_title="Date",
            yaxis_title="Predicted Cases",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def compare_intervention_scenarios(self, scenarios):
        """Compare multiple intervention scenarios"""
        results = {}
        
        for scenario_name, interventions in scenarios.items():
            results[scenario_name] = self.simulate_interventions(interventions)
        
        # Create comparison dataframe
        comparison = pd.DataFrame(results).T
        
        return comparison
    
    def get_cost_benefit_analysis(self, intervention_costs):
        """Perform cost-benefit analysis"""
        if self.intervention_cases is None:
            return None
        
        analysis = {}
        
        for intervention, cost in intervention_costs.items():
            # Simulate single intervention
            results = self.simulate_interventions([intervention])
            
            # Calculate cost per case averted
            cost_per_case = cost / results['cases_averted'] if results['cases_averted'] > 0 else float('inf')
            
            analysis[intervention] = {
                'cost': cost,
                'cases_averted': results['cases_averted'],
                'cost_per_case': cost_per_case,
                'peak_reduction': results['peak_reduction'],
                'peak_delay': results['peak_delay']
            }
        
        return analysis