import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class EpidemicModels:
    """Epidemic modeling and forecasting"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.predictions = {}
        
    def sir_model(self, region_data, days=30):
        """
        SIR Model: Susceptible-Infectious-Recovered
        """
        # Extract region-specific data
        region_data = region_data.sort_values('date')
        
        # Initial conditions
        N = region_data['population'].iloc[0]  # Total population
        I0 = region_data['confirmed'].iloc[-1]  # Initial infected
        R0 = region_data['recovered'].iloc[-1] + region_data['deaths'].iloc[-1]  # Initial recovered
        S0 = N - I0 - R0  # Initial susceptible
        
        # Estimate parameters from data
        recent_growth = region_data['growth_rate'].iloc[-7:].mean() / 100
        beta = recent_growth + 0.3  # Transmission rate
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
            'peak_date': pred_dates[np.argmax(I)],
            'peak_cases': np.max(I)
        })
        
        return predictions
    
    def seir_model(self, region_data, days=30):
        """
        SEIR Model: Susceptible-Exposed-Infectious-Recovered
        """
        region_data = region_data.sort_values('date')
        
        N = region_data['population'].iloc[0]
        E0 = region_data['new_cases'].iloc[-7:].mean() * 5  # Estimate exposed
        I0 = region_data['confirmed'].iloc[-1]
        R0 = region_data['recovered'].iloc[-1] + region_data['deaths'].iloc[-1]
        S0 = N - E0 - I0 - R0
        
        # Parameters
        beta = 0.5  # Transmission rate
        sigma = 0.2  # Incubation rate (1/latent period)
        gamma = 0.1  # Recovery rate
        
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
            'peak_date': pred_dates[np.argmax(I)],
            'peak_cases': np.max(I)
        })
        
        return predictions
    
    def prophet_forecast(self, region_data, days=30):
        """
        Facebook Prophet for time series forecasting
        """
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': region_data['date'],
            'y': region_data['confirmed']
        })
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=days)
        
        # Predict
        forecast = model.predict(future)
        
        # Extract predictions
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
        predictions.columns = ['date', 'predicted_cases', 'lower_bound', 'upper_bound']
        
        # Add additional columns
        predictions['new_cases'] = predictions['predicted_cases'].diff().fillna(0)
        predictions['peak_date'] = predictions.loc[predictions['predicted_cases'].idxmax(), 'date']
        predictions['peak_cases'] = predictions['predicted_cases'].max()
        
        return predictions
    
    def arima_forecast(self, region_data, days=30):
        """
        ARIMA model for time series forecasting
        """
        # Prepare time series
        ts = region_data.set_index('date')['confirmed']
        
        # Fit ARIMA model
        try:
            model = ARIMA(ts, order=(5,1,0))
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
                'predicted_cases': pred_series.values,
                'new_cases': pred_series.diff().fillna(0).values
            })
            
            predictions['peak_date'] = predictions.loc[predictions['predicted_cases'].idxmax(), 'date']
            predictions['peak_cases'] = predictions['predicted_cases'].max()
            
        except Exception as e:
            st.warning(f"ARIMA model failed, using simple exponential smoothing: {str(e)}")
            # Fallback to simple exponential smoothing
            alpha = 0.3
            last_value = ts.iloc[-1]
            predictions = []
            dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=days)
            
            for i, date in enumerate(dates):
                if i == 0:
                    pred = last_value
                else:
                    pred = predictions[-1]['predicted_cases'] * (1 + np.random.normal(0, 0.02))
                
                predictions.append({
                    'date': date,
                    'predicted_cases': pred,
                    'new_cases': pred - (predictions[-1]['predicted_cases'] if i > 0 else last_value)
                })
            
            predictions = pd.DataFrame(predictions)
            predictions['peak_date'] = predictions.loc[predictions['predicted_cases'].idxmax(), 'date']
            predictions['peak_cases'] = predictions['predicted_cases'].max()
        
        return predictions
    
    def calculate_r0(self, region_data):
        """Calculate reproduction number"""
        # Simple R0 calculation using growth rate and serial interval
        serial_interval = 5  # Average days between cases
        growth_rate = region_data['growth_rate'].iloc[-7:].mean() / 100
        r0 = 1 + growth_rate * serial_interval
        
        return max(0, r0)
    
    def calculate_herd_immunity_threshold(self, r0):
        """Calculate herd immunity threshold"""
        if r0 > 1:
            return 1 - 1/r0
        return 0
    
    def estimate_peak(self, region_data):
        """Estimate peak of epidemic"""
        # Fit a quadratic to recent data
        recent_cases = region_data['confirmed'].tail(14).values
        x = np.arange(len(recent_cases))
        
        try:
            # Fit quadratic
            coeffs = np.polyfit(x, recent_cases, 2)
            
            # Find vertex
            if coeffs[0] < 0:  # Concave down (peak)
                peak_x = -coeffs[1] / (2 * coeffs[0])
                peak_y = np.polyval(coeffs, peak_x)
                
                if 0 <= peak_x <= len(recent_cases) * 2:
                    days_to_peak = peak_x - (len(recent_cases) - 1)
                    return days_to_peak, peak_y
        except:
            pass
        
        return None, None