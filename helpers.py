import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def format_number(num):
    """Format large numbers with commas"""
    if num is None:
        return "N/A"
    try:
        return f"{int(num):,}"
    except:
        return str(num)

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for time series"""
    if len(data) < 2:
        return None, None
    
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    margin = z * (std / np.sqrt(n))
    
    return mean - margin, mean + margin

def validate_date_range(start_date, end_date):
    """Validate date range input"""
    if not start_date or not end_date:
        return False, "Please select both start and end dates"
    
    if start_date > end_date:
        return False, "Start date must be before end date"
    
    if end_date > datetime.now().date():
        return False, "End date cannot be in the future"
    
    return True, "Valid date range"

def calculate_doubling_time(growth_rates):
    """Calculate doubling time from growth rates"""
    avg_growth = np.mean(growth_rates) / 100  # Convert to decimal
    
    if avg_growth <= 0:
        return float('inf')
    
    doubling_time = np.log(2) / np.log(1 + avg_growth)
    return doubling_time

def moving_average(data, window=7):
    """Calculate moving average"""
    return data.rolling(window=window, min_periods=1).mean()

def detect_outliers(data, threshold=3):
    """Detect outliers using z-score method"""
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

def calculate_epidemic_phase(growth_rate, r0):
    """Determine epidemic phase based on metrics"""
    if r0 > 1.5 and growth_rate > 10:
        return "Exponential Growth"
    elif r0 > 1 and growth_rate > 0:
        return "Growth Phase"
    elif r0 < 1 and growth_rate < 0:
        return "Declining Phase"
    else:
        return "Plateau/Stable"

def get_seasonal_factor(month):
    """Get seasonal transmission factor based on month"""
    # Northern hemisphere assumptions
    seasonal_factors = {
        12: 1.3, 1: 1.4, 2: 1.3,  # Winter
        3: 1.1, 4: 1.0, 5: 0.9,    # Spring
        6: 0.8, 7: 0.7, 8: 0.8,    # Summer
        9: 0.9, 10: 1.0, 11: 1.1   # Fall
    }
    return seasonal_factors.get(month, 1.0)

def calculate_healthcare_capacity(icu_beds, population, cases):
    """Calculate healthcare system capacity strain"""
    icu_per_100k = (icu_beds / population) * 100000
    cases_per_100k = (cases / population) * 100000
    
    # Estimate ICU need (typically 5-10% of cases need ICU)
    icu_need = cases * 0.05
    icu_occupancy = (icu_need / icu_beds) * 100 if icu_beds > 0 else float('inf')
    
    return {
        'icu_beds_per_100k': icu_per_100k,
        'cases_per_100k': cases_per_100k,
        'estimated_icu_need': icu_need,
        'icu_occupancy_rate': min(icu_occupancy, 100),
        'capacity_status': 'Critical' if icu_occupancy > 90 else 'Strained' if icu_occupancy > 70 else 'Adequate'
    }