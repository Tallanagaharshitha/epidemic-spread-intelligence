import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class EpidemicModels:
    """Epidemic modeling and forecasting without prophet dependency"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.predictions = {}
        
    def sir_model(self, region_data, days=30):
        """
        SIR Model: Susceptible-Infectious-Recovered
        """
        try:
            # Extract region-specific data
            region_data = region_data.sort_values('date')
            
            # Initial conditions
            N = region_data['population'].iloc[0] if len(region_data) > 0 else 1000000
            I0 = region_data['confirmed'].iloc[-1] if len(region_data) > 0 else 100
            R0 = region_data['recovered'].iloc[-1] + region_data['deaths'].iloc[-1] if len(region_data) > 0 else 0
            S0 = N - I0 - R0
            
            # Estimate parameters from data
            recent_growth = region_data['growth_rate'].iloc[-7:].mean() / 100 if len(region_data) >= 7 else 0.05
            beta = min(max(recent_growth + 0.3, 0.1), 1.0)  # Transmission rate
            gamma = 0.1  # Recovery rate
            
            # Time vector
            t = np.linspace(0, days, days)
            
            # SIR differential equations
            def deriv(y, t, N, beta, gamma):
                S, I, R = y
                dSdt = -beta * S * I / N
                dIdt = beta * S * I / N - gamma * I
                dRdt = gamma * I
                return dSdt, dIdt, dRdt
            
            # Initial conditions vector
            y0 = S0, I0, R0
            
            # Integrate ODEs
            ret = odeint(deriv, y0, t, args=(N, beta, gamma))
            S, I, R = ret.T
            
            # Create prediction dataframe
            last_date = region_data['date'].max()
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            predictions = pd.DataFrame({
                'date': pred_dates,
                'susceptible': S,
                'infected': I,
                'recovered': R,
                'predicted_cases': I + R,
                'new_cases': np.diff(I, prepend=I[0]),
                'peak_date': pred_dates[np.argmax(I)] if len(I) > 0 else pred_dates[0],
                'peak_cases': np.max(I) if len(I) > 0 else 0
            })
            
            return predictions
        except Exception as e:
            st.warning(f"SIR model error: {str(e)}. Using fallback.")
            return self._fallback_forecast(region_data, days)
    
    def seir_model(self, region_data, days=30):
        """
        SEIR Model: Susceptible-Exposed-Infectious-Recovered
        """
        try:
            region_data = region_data.sort_values('date')
            
            N = region_data['population'].iloc[0] if len(region_data) > 0 else 1000000
            E0 = region_data['new_cases'].iloc[-7:].mean() * 5 if len(region_data) >= 7 else 100
            I0 = region_data['confirmed'].iloc[-1] if len(region_data) > 0 else 100
            R0 = region_data['recovered'].iloc[-1] + region_data['deaths'].iloc[-1] if len(region_data) > 0 else 0
            S0 = N - E0 - I0 - R0
            
            # Parameters
            beta = 0.5
            sigma = 0.2
            gamma = 0.1
            
            t = np.linspace(0, days, days)
            
            def deriv(y, t, N, beta, sigma, gamma):
                S, E, I, R = y
                dSdt = -beta * S * I / N
                dEdt = beta * S * I / N - sigma * E
                dIdt = sigma * E - gamma * I
                dRdt = gamma * I
                return dSdt, dEdt, dIdt, dRdt
            
            y0 = S0, E0, I0, R0
            ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
            S, E, I, R = ret.T
            
            last_date = region_data['date'].max()
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            predictions = pd.DataFrame({
                'date': pred_dates,
                'susceptible': S,
                'exposed': E,
                'infected': I,
                'recovered': R,
                'predicted_cases': I + R,
                'new_cases': np.diff(I, prepend=I[0]),
                'peak_date': pred_dates[np.argmax(I)] if len(I) > 0 else pred_dates[0],
                'peak_cases': np.max(I) if len(I) > 0 else 0
            })
            
            return predictions
        except Exception as e:
            st.warning(f"SEIR model error: {str(e)}. Using fallback.")
            return self._fallback_forecast(region_data, days)
    
    def arima_forecast(self, region_data, days=30):
        """
        ARIMA model for time series forecasting
        """
        try:
            # Prepare time series
            ts = region_data.set_index('date')['confirmed']
            
            # Fit ARIMA model
            model = ARIMA(ts, order=(3, 1, 2))
            model_fit = model.fit()
            
            # Make predictions
            forecast = model_fit.forecast(steps=days)
            pred_series = pd.Series(forecast, index=pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1), 
                periods=days
            ))
            
            # Create prediction dataframe
            predictions = pd.DataFrame({
                'date': pred_series.index,
                'predicted_cases': np.maximum(pred_series.values, 0),
                'new_cases': np.maximum(pred_series.diff().fillna(0).values, 0)
            })
            
            predictions['peak_date'] = predictions.loc[predictions['predicted_cases'].idxmax(), 'date'] if len(predictions) > 0 else predictions['date'].iloc[-1]
            predictions['peak_cases'] = predictions['predicted_cases'].max() if len(predictions) > 0 else 0
            
            return predictions
            
        except Exception as e:
            st.warning(f"ARIMA model failed: {str(e)}. Using exponential smoothing.")
            return self._exponential_smoothing_forecast(region_data, days)
    
    def _exponential_smoothing_forecast(self, region_data, days=30):
        """Fallback using exponential smoothing"""
        try:
            ts = region_data.set_index('date')['confirmed']
            
            # Use simple exponential smoothing
            model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            model_fit = model.fit()
            
            forecast = model_fit.forecast(steps=days)
            
            pred_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=days)
            
            predictions = pd.DataFrame({
                'date': pred_dates,
                'predicted_cases': np.maximum(forecast.values, 0),
                'new_cases': np.maximum(np.diff(np.concatenate([[ts.iloc[-1]], forecast.values])), 0)
            })
            
            predictions['peak_date'] = predictions.loc[predictions['predicted_cases'].idxmax(), 'date'] if len(predictions) > 0 else predictions['date'].iloc[-1]
            predictions['peak_cases'] = predictions['predicted_cases'].max() if len(predictions) > 0 else 0
            
            return predictions
        except Exception as e:
            return self._simple_growth_forecast(region_data, days)
    
    def _simple_growth_forecast(self, region_data, days=30):
        """Ultimate fallback using simple growth model"""
        last_value = region_data['confirmed'].iloc[-1] if len(region_data) > 0 else 100
        growth_rate = region_data['growth_rate'].iloc[-7:].mean() / 100 if len(region_data) >= 7 else 0.03
        
        predictions = []
        pred_dates = pd.date_range(start=region_data['date'].max() + pd.Timedelta(days=1), periods=days)
        
        current = last_value
        for i, date in enumerate(pred_dates):
            current = current * (1 + growth_rate * (1 - i/days * 0.5))
            predictions.append({
                'date': date,
                'predicted_cases': current,
                'new_cases': current * growth_rate
            })
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df['peak_date'] = predictions_df.loc[predictions_df['predicted_cases'].idxmax(), 'date']
        predictions_df['peak_cases'] = predictions_df['predicted_cases'].max()
        
        return predictions_df
    
    def _fallback_forecast(self, region_data, days=30):
        """Simple linear fallback forecast"""
        last_date = region_data['date'].max()
        last_value = region_data['confirmed'].iloc[-1] if len(region_data) > 0 else 100
        
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        # Simple linear growth then decay
        predictions = []
        for i, date in enumerate(pred_dates):
            if i < days * 0.6:
                growth = 1 + (i / (days * 0.6)) * 0.3
            else:
                growth = 1.3 - ((i - days * 0.6) / (days * 0.4)) * 0.5
            predicted = last_value * growth
            predictions.append({
                'date': date,
                'predicted_cases': max(0, predicted),
                'new_cases': max(0, predicted * 0.05)
            })
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df['peak_date'] = predictions_df.loc[predictions_df['predicted_cases'].idxmax(), 'date']
        predictions_df['peak_cases'] = predictions_df['predicted_cases'].max()
        
        return predictions_df
    
    def prophet_forecast(self, region_data, days=30):
        """Alias for ARIMA (replacement for prophet)"""
        return self.arima_forecast(region_data, days)
    
    def calculate_r0(self, region_data):
        """Calculate reproduction number"""
        try:
            # Simple R0 calculation using growth rate and serial interval
            serial_interval = 5  # Average days between cases
            growth_rate = region_data['growth_rate'].iloc[-7:].mean() / 100 if len(region_data) >= 7 else 0
            r0 = 1 + growth_rate * serial_interval
            return max(0.5, min(3.0, r0))  # Clamp between 0.5 and 3.0
        except:
            return 1.2
    
    def calculate_herd_immunity_threshold(self, r0):
        """Calculate herd immunity threshold"""
        if r0 > 1:
            return 1 - 1/r0
        return 0
    
    def estimate_peak(self, region_data):
        """Estimate peak of epidemic"""
        try:
            recent_cases = region_data['confirmed'].tail(14).values
            x = np.arange(len(recent_cases))
            
            # Fit quadratic
            coeffs = np.polyfit(x, recent_cases, 2)
            
            # Find vertex
            if coeffs[0] < 0:
                peak_x = -coeffs[1] / (2 * coeffs[0])
                peak_y = np.polyval(coeffs, peak_x)
                
                if 0 <= peak_x <= len(recent_cases) * 2:
                    days_to_peak = peak_x - (len(recent_cases) - 1)
                    return days_to_peak, peak_y
            
            return None, None
        except:
            return None, None
