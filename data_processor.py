import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataProcessor:
    """Handle data loading, cleaning, and feature engineering"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def load_csv(self, file):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file)
            return self.validate_and_clean(df)
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None
    
    def load_excel(self, file):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file)
            return self.validate_and_clean(df)
        except Exception as e:
            st.error(f"Error loading Excel: {str(e)}")
            return None
    
    def validate_and_clean(self, df):
        """Validate and clean the dataframe"""
        # Expected columns
        expected_cols = ['date', 'region', 'confirmed', 'recovered', 'deaths', 'population']
        
        # Check if required columns exist
        df.columns = df.columns.str.lower().str.strip()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate derived features
        df = self.engineer_features(df)
        
        self.processed_data = df
        return df
    
    def engineer_features(self, df):
        """Engineer additional features"""
        # Sort by date
        df = df.sort_values(['region', 'date'])
        
        # Calculate daily new cases
        df['new_cases'] = df.groupby('region')['confirmed'].diff().fillna(0)
        
        # Calculate growth rate
        df['growth_rate'] = df.groupby('region')['new_cases'].pct_change().fillna(0) * 100
        
        # Calculate 7-day moving average
        df['ma_7d'] = df.groupby('region')['new_cases'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Calculate reproduction number (simplified R0)
        df['r0_estimate'] = df.groupby('region')['growth_rate'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean() / 100 + 1
        )
        
        # Calculate case fatality rate
        df['cfr'] = (df['deaths'] / df['confirmed'] * 100).fillna(0)
        
        # Calculate incidence per 100,000
        df['incidence_per_100k'] = (df['confirmed'] / df['population'] * 100000).fillna(0)
        
        # Calculate doubling time (days)
        df['doubling_time'] = np.log(2) / np.log(1 + df['growth_rate']/100).fillna(0)
        df['doubling_time'] = df['doubling_time'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def get_summary_stats(self, region=None):
        """Get summary statistics"""
        if region:
            df = self.processed_data[self.processed_data['region'] == region]
        else:
            df = self.processed_data
            
        latest = df[df['date'] == df['date'].max()]
        
        stats = {
            'total_confirmed': int(latest['confirmed'].sum()),
            'total_recovered': int(latest['recovered'].sum()),
            'total_deaths': int(latest['deaths'].sum()),
            'active_cases': int(latest['confirmed'].sum() - latest['recovered'].sum() - latest['deaths'].sum()),
            'cfr': (latest['deaths'].sum() / latest['confirmed'].sum() * 100) if latest['confirmed'].sum() > 0 else 0,
            'recovery_rate': (latest['recovered'].sum() / latest['confirmed'].sum() * 100) if latest['confirmed'].sum() > 0 else 0,
            'regions_affected': latest['region'].nunique(),
            'latest_date': df['date'].max().strftime('%Y-%m-%d'),
            'total_population': int(latest['population'].sum())
        }
        
        return stats
    
    def get_time_series(self, region=None):
        """Get time series data"""
        if region:
            return self.processed_data[self.processed_data['region'] == region]
        return self.processed_data