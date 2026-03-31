import pandas as pd
import numpy as np
from datetime import datetime
import io
from fpdf import FPDF
import streamlit as st

class ReportGenerator:
    """Generate reports in various formats"""
    
    def __init__(self):
        self.pdf = None
        
    def generate_pdf_report(self, data, predictions, stats, recommendations=None):
        """Generate PDF report"""
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Epidemic Spread Intelligence Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Date
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
        pdf.ln(10)
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        summary = f"""
        As of {stats.get('latest_date', 'the latest data')}, there are {stats.get('total_confirmed', 0):,} confirmed cases 
        across {stats.get('regions_affected', 0)} regions. The case fatality rate is {stats.get('cfr', 0):.1f}% 
        with {stats.get('active_cases', 0):,} active cases.
        """
        pdf.multi_cell(0, 5, summary)
        pdf.ln(5)
        
        # Key Statistics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Key Statistics', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        stats_text = f"""
        Total Confirmed: {stats.get('total_confirmed', 0):,}
        Total Recovered: {stats.get('total_recovered', 0):,}
        Total Deaths: {stats.get('total_deaths', 0):,}
        Active Cases: {stats.get('active_cases', 0):,}
        Case Fatality Rate: {stats.get('cfr', 0):.1f}%
        Recovery Rate: {stats.get('recovery_rate', 0):.1f}%
        Regions Affected: {stats.get('regions_affected', 0)}
        """
        pdf.multi_cell(0, 5, stats_text)
        pdf.ln(5)
        
        # Predictions
        if predictions is not None:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Predictions Summary', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for region in predictions['region'].unique():
                region_pred = predictions[predictions['region'] == region]
                peak_date = region_pred['peak_date'].iloc[0] if 'peak_date' in region_pred.columns else 'Unknown'
                peak_cases = region_pred['peak_cases'].iloc[0] if 'peak_cases' in region_pred.columns else 0
                
                pdf.cell(0, 5, f"{region}: Peak predicted on {peak_date} with {peak_cases:,.0f} cases", 0, 1)
        
        # Recommendations
        if recommendations:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Policy Recommendations', 0, 1)
            pdf.ln(5)
            
            # Summary
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Executive Summary:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, recommendations.get('summary', 'No summary available'))
            pdf.ln(5)
            
            # Short-term
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Short-term Actions:', 0, 1)
            pdf.set_font('Arial', '', 10)
            for action in recommendations.get('short_term', []):
                pdf.cell(0, 5, f"• {action}", 0, 1)
            pdf.ln(5)
            
            # Long-term
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Long-term Strategy:', 0, 1)
            pdf.set_font('Arial', '', 10)
            for strategy in recommendations.get('long_term', []):
                pdf.cell(0, 5, f"• {strategy}", 0, 1)
            pdf.ln(5)
            
            # Risk Assessment
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Risk Assessment:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, recommendations.get('risk_assessment', 'No risk assessment available'))
        
        # Output PDF
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return io.BytesIO(pdf_bytes)
    
    def generate_excel_report(self, data, predictions):
        """Generate Excel report with multiple sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Raw data
            data.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Summary statistics
            summary = data.groupby('region').agg({
                'confirmed': 'max',
                'deaths': 'max',
                'recovered': 'max',
                'cfr': 'last'
            }).reset_index()
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Time series
            pivot = data.pivot_table(
                values='confirmed',
                index='date',
                columns='region',
                aggfunc='sum'
            ).fillna(0)
            pivot.to_excel(writer, sheet_name='Time Series')
            
            # Predictions
            if predictions is not None:
                predictions.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Growth rates
            growth = data.pivot_table(
                values='growth_rate',
                index='date',
                columns='region',
                aggfunc='mean'
            ).fillna(0)
            growth.to_excel(writer, sheet_name='Growth Rates')
            
            # Format Excel
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:Z', 15)
        
        output.seek(0)
        return output
    
    def generate_html_report(self, data, predictions, stats):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Epidemic Intelligence Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #1f77b4; }}
                h2 {{ color: #2c3e50; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #1f77b4; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .stats-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
                .stat-card {{ background-color: #f8f9fa; border-radius: 10px; padding: 20px; 
                              flex: 1; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
                .stat-label {{ color: #666; font-size: 14px; }}
                .footer {{ margin-top: 50px; color: #666; font-size: 12px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>🦠 Epidemic Spread Intelligence Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Key Statistics</h2>
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-value">{stats.get('total_confirmed', 0):,}</div>
                    <div class="stat-label">Total Confirmed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('active_cases', 0):,}</div>
                    <div class="stat-label">Active Cases</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('cfr', 0):.1f}%</div>
                    <div class="stat-label">Case Fatality Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats.get('regions_affected', 0)}</div>
                    <div class="stat-label">Regions Affected</div>
                </div>
            </div>
            
            <h2>Regional Summary</h2>
            <table>
                <tr>
                    <th>Region</th>
                    <th>Total Cases</th>
                    <th>Deaths</th>
                    <th>Recovered</th>
                    <th>CFR (%)</th>
                </tr>
        """
        
        # Add regional data
        latest = data[data['date'] == data['date'].max()]
        for _, row in latest.iterrows():
            html += f"""
                <tr>
                    <td>{row['region']}</td>
                    <td>{row['confirmed']:,}</td>
                    <td>{row['deaths']:,}</td>
                    <td>{row['recovered']:,}</td>
                    <td>{row['cfr']:.1f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Time Series Preview</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Region</th>
                    <th>Confirmed</th>
                    <th>New Cases</th>
                    <th>Growth Rate (%)</th>
                </tr>
        """
        
        # Add time series preview (last 10 rows)
        for _, row in data.tail(10).iterrows():
            html += f"""
                <tr>
                    <td>{row['date'].strftime('%Y-%m-%d')}</td>
                    <td>{row['region']}</td>
                    <td>{row['confirmed']:,}</td>
                    <td>{row['new_cases']:,}</td>
                    <td>{row['growth_rate']:.1f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="footer">
                <p>This report was automatically generated by Epidemic Spread Intelligence System.</p>
                <p>For more detailed analysis, please refer to the interactive dashboard.</p>
            </div>
        </body>
        </html>
        """
        
        return html.encode('utf-8')