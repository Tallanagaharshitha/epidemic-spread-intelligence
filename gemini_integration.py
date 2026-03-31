import google.generativeai as genai
import streamlit as st
import json
from datetime import datetime

class GeminiAnalyzer:
    """Integration with Google Gemini 1.5 Flash API"""
    
    def __init__(self,api_key):
        """Initialize Gemini with API key"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            self.model = None
    
    def generate_policy_recommendations(self, stats, predictions, interventions=None):
        """Generate policy recommendations based on data"""
        if not self.model:
            return self._get_fallback_recommendations(stats)
        
        # Prepare context for Gemini
        context = self._prepare_context(stats, predictions, interventions)
        
        prompt = f"""
        You are an expert epidemiologist and public health policy advisor. 
        Based on the following epidemic data, provide comprehensive policy recommendations.
        
        EPIDEMIC DATA:
        {context}
        
        Please provide:
        1. Executive Summary (2-3 sentences)
        2. Short-term actions (next 30 days) - 5 bullet points
        3. Long-term strategy (3-6 months) - 5 bullet points
        4. Risk Assessment (potential challenges)
        5. Regional-specific recommendations
        
        Format the response as a JSON object with keys:
        - summary
        - short_term (list)
        - long_term (list)
        - risk_assessment
        - regional (object with region names as keys)
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            st.warning(f"Gemini API error: {str(e)}. Using fallback recommendations.")
            return self._get_fallback_recommendations(stats)
    
    def answer_query(self, query, context_data):
        """Answer user queries about the epidemic"""
        if not self.model:
            return "Gemini API not configured. Please check your API key."
        
        prompt = f"""
        Based on the following epidemic data, answer the user's question.
        
        EPIDEMIC DATA:
        {context_data}
        
        USER QUESTION: {query}
        
        Provide a clear, informative answer with specific data references when possible.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def explain_predictions(self, predictions_data):
        """Generate natural language explanations of predictions"""
        if not self.model:
            return "Gemini API not configured."
        
        prompt = f"""
        Explain the following epidemic predictions in simple terms:
        
        {predictions_data}
        
        Provide:
        1. What the numbers mean
        2. Key trends to watch
        3. Confidence level interpretation
        4. What factors might change these predictions
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _prepare_context(self, stats, predictions, interventions):
        """Prepare context data for Gemini"""
        context = f"""
        Current Statistics (as of {stats.get('latest_date', 'recent')}):
        - Total Confirmed Cases: {stats.get('total_confirmed', 'N/A'):,}
        - Active Cases: {stats.get('active_cases', 'N/A'):,}
        - Total Recovered: {stats.get('total_recovered', 'N/A'):,}
        - Total Deaths: {stats.get('total_deaths', 'N/A'):,}
        - Case Fatality Rate: {stats.get('cfr', 0):.1f}%
        - Recovery Rate: {stats.get('recovery_rate', 0):.1f}%
        - Regions Affected: {stats.get('regions_affected', 'N/A')}
        
        Predictions Summary:
        """
        
        if predictions is not None and not predictions.empty:
            for region in predictions['region'].unique():
                region_pred = predictions[predictions['region'] == region]
                peak_date = region_pred['peak_date'].iloc[0] if 'peak_date' in region_pred.columns else 'Unknown'
                peak_cases = region_pred['peak_cases'].iloc[0] if 'peak_cases' in region_pred.columns else 'Unknown'
                
                context += f"""
                - {region}: Peak predicted on {peak_date} with {peak_cases:,.0f} cases
                """
        
        if interventions:
            context += f"""
            
            Intervention Simulation Results:
            - Cases Averted: {interventions.get('cases_averted', 'N/A'):,}
            - Peak Reduction: {interventions.get('peak_reduction', 'N/A')}%
            - Delay in Peak: {interventions.get('peak_delay', 'N/A')} days
            """
        
        return context
    
    def _parse_response(self, response_text):
        """Parse Gemini response into structured format"""
        try:
            # Try to parse as JSON
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
        except:
            pass
        
        # Fallback: Parse text into structure
        lines = response_text.split('\n')
        recommendations = {
            'summary': '',
            'short_term': [],
            'long_term': [],
            'risk_assessment': '',
            'regional': {}
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            
            if 'summary' in lower_line or 'executive' in lower_line:
                current_section = 'summary'
            elif 'short-term' in lower_line or 'short term' in lower_line:
                current_section = 'short_term'
            elif 'long-term' in lower_line or 'long term' in lower_line:
                current_section = 'long_term'
            elif 'risk' in lower_line:
                current_section = 'risk_assessment'
            elif 'regional' in lower_line:
                current_section = 'regional'
            elif current_section == 'summary':
                recommendations['summary'] += ' ' + line
            elif current_section == 'short_term' and line.startswith(('•', '-', '*')):
                recommendations['short_term'].append(line.lstrip('•-* '))
            elif current_section == 'long_term' and line.startswith(('•', '-', '*')):
                recommendations['long_term'].append(line.lstrip('•-* '))
            elif current_section == 'risk_assessment':
                recommendations['risk_assessment'] += ' ' + line
        
        return recommendations
    
    def _get_fallback_recommendations(self, stats):
        """Provide fallback recommendations when Gemini is unavailable"""
        cfr = stats.get('cfr', 0)
        active = stats.get('active_cases', 0)
        
        return {
            'summary': f"Based on current data with {active:,} active cases and {cfr:.1f}% CFR, immediate public health interventions are recommended.",
            
            'short_term': [
                "Increase testing capacity in affected regions",
                "Implement targeted isolation measures for high-risk areas",
                "Enhance contact tracing efforts",
                "Deploy mobile testing units to underserved areas",
                "Launch public awareness campaign about preventive measures"
            ],
            
            'long_term': [
                "Strengthen healthcare infrastructure",
                "Develop vaccination distribution plan",
                "Establish early warning surveillance system",
                "Create stockpile of essential medical supplies",
                "Train healthcare workers in epidemic response"
            ],
            
            'risk_assessment': "High risk of continued spread without intervention. Healthcare system capacity may be strained if current trends continue.",
            
            'regional': {
                'All Regions': "Implement unified response strategy while allowing local adaptation based on specific needs."
            }
        }