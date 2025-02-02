import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error , mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from mango import scheduler, Tuner
import os

# Specify the directory 
FIGURE_DIR = r'C:\DATA\Job\Causalens\Figures'
output_dir = r"C:\DATA\Job\Causalens\Analysis"
# Create the directory if it doesn't exist
os.makedirs(FIGURE_DIR, exist_ok=True)  

def Forcast_daily_distribution(daily_start_station_stats, daily_end_station_stats, station_identifier,Arima_Tuner =  False, by_id=True):
    global train_data
    
    if Arima_Tuner:
        print("Arima Tuner is active.")
    else:
        print("Arima Tuner is not active.")    

    daily_start_station_stats['start_station_id'] = pd.to_numeric(daily_start_station_stats['start_station_id'], errors='coerce')
    daily_end_station_stats['end_station_id'] = pd.to_numeric(daily_end_station_stats['end_station_id'], errors='coerce')

    daily_start_station_stats['day'] = pd.to_datetime(daily_start_station_stats['day'], format='%d/%m/%Y')
    daily_end_station_stats['day'] = pd.to_datetime(daily_end_station_stats['day'], format='%d/%m/%Y')

    specific_station = daily_start_station_stats[daily_start_station_stats['start_station_id'] == station_identifier]
    the_station_name = specific_station['start_station_name'].values[0]
    
    print ('The station name is :',the_station_name)

    if by_id:
        # Filter based on station ID
        start_data = daily_start_station_stats[daily_start_station_stats['start_station_id'] == station_identifier]
        end_data = daily_end_station_stats[daily_end_station_stats['end_station_id'] == station_identifier]
    else:
        # Filter based on station name
        start_data = daily_start_station_stats[daily_start_station_stats['start_station_name'] == station_identifier]
        end_data = daily_end_station_stats[daily_end_station_stats['end_station_name'] == station_identifier]

    # Check if filtering worked
    if start_data.empty or end_data.empty:
        print(f"No data found for station identifier: {station_identifier}")
        return

    # Group by 'day' and sum total rides to avoid duplicate indices
    start_data = start_data.groupby('day', as_index=False)['departure'].sum()
    end_data = end_data.groupby('day', as_index=False)['arrival'].sum()

    # Set 'day' as the index
    start_data.set_index('day', inplace=True)
    end_data.set_index('day', inplace=True)
    
    # Create a date range that covers all days in both DataFrames
    all_days = pd.date_range(start=min(start_data.index.min(), end_data.index.min()), 
                             end=max(start_data.index.max(), end_data.index.max()))

    # Reindex each DataFrame to include all days, filling missing values with 0
    start_data = start_data.reindex(all_days, fill_value=0)
    end_data = end_data.reindex(all_days, fill_value=0)

    # Calculate the difference
    Diff = end_data['arrival'] - start_data['departure']

    # Create the final DataFrame
    result_data = pd.DataFrame({
        'day': start_data.index,
        'total_rides_start': start_data['departure'],
        'total_rides_end': end_data['arrival'],
        'diff': Diff
    })

    station_name = f"Station_{station_identifier}"  
    output_filename = os.path.join(output_dir, f'{station_name}.csv')
    result_data.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

    # Create time series for the ARIMA model
    time_series = pd.Series(Diff, index=start_data.index)
       
    train_size = int(len(time_series) * 11 / 12)
    train_data = time_series.iloc[:train_size]
    test_data = time_series.iloc[train_size:]
        
    # Check stationarity and fit ARIMA model
    test_stationarity(time_series)

    plot_net_demand(time_series,station_identifier,the_station_name)
    
    # Call decomposing and fitting functions
    decompose_time_series(time_series,station_identifier,the_station_name)

    if (Arima_Tuner):
        param_space = dict(p= range(0, 30),
                       d= range(0, 2),
                       q =range(0, 30),
                       trend = ['n', 'c', 't', 'ct']
                      )

        conf_Dict = dict()
        conf_Dict['num_iteration'] = 100
        tuner = Tuner(param_space, arima_tuner_function, conf_Dict)
        results = tuner.minimize()
        print('best parameters:', results['best_params'])
        print('best loss:', results['best_objective'])
        best_order = (results['best_params']['d'], results['best_params']['p'], results['best_params']['q'])

        # Extract parameters and loss
        best_params = results['best_params']
        best_loss = results['best_objective']

        # Create a DataFrame
        results_df = pd.DataFrame({
           'p': [best_params['p']],
            'd': [best_params['d']],
            'q': [best_params['q']],
            'trend': [best_params['trend']],
            'best_loss': [best_loss]
        })

        # Save the DataFrame to a CSV file in the specified directory
        results_df.to_csv(os.path.join(output_dir, 'arima_tuning_results.csv'), index=False)
    
    # Read the CSV file into a DataFrame
    results_df = pd.read_csv(os.path.join(output_dir, 'arima_tuning_results.csv'))

    # p, d, q = results['best_params']['d'], results['best_params']['p'], results['best_params']['q']
    # Read values of p, d, q, and trend from the DataFrame
    p, d, q, trend = results_df['p'].iloc[0], results_df['d'].iloc[0], results_df['q'].iloc[0], results_df['trend'].iloc[0]

    # Print all values in one line
    print(f'Optimal ARIMA parameters: p: {p}, d: {d}, q: {q}, trend: {trend}')
    
    # Fit the ARIMA model to the training data
    #p, d, q = 29, 0, 29
    forecast_and_evaluate_ARIMA(train_data, test_data, p, d, q, station_identifier,the_station_name, trend=None)

def decompose_time_series(time_series,station_identifier,the_station_name):
    decomposition = seasonal_decompose(time_series, model='additive', period=30)  #7 or 30
    
    # Plot the decomposed components
    plt.figure(figsize=(10, 6))
    
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Observed')
    plt.title(f'{the_station_name}: Observed Time Series')
    plt.legend(loc='best')
    
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend', color='r')
    plt.title(f'{the_station_name}: Trend')
    plt.legend(loc='best')
    
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality', color='g')
    plt.title(f'{the_station_name}: Seasonality')
    plt.legend(loc='best')
    
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals (Noise)', color='b')
    plt.title(f'{the_station_name}: Residuals (Noise)')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'Decompose_the_time_series_net_ride_{station_identifier}_{the_station_name}.png'))
    plt.show()

def test_stationarity(time_series):
    adf_result = adfuller(time_series)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])

def arima_tuner_function(args_list):
    global train_data
    
    Error_MSE = False 
    Error_MAE = True

    params_evaluated = []
    results = []
    
    for params in args_list:
        try:
            p,d,q = params['p'],params['d'], params['q']
            trend = params['trend']
            
            model = ARIMA(train_data, order=(p,d,q))#, trend=trend)
            predictions = model.fit()
            if (Error_MSE): mse = mean_squared_error(train_data, predictions.fittedvalues)   
            if (Error_MAE): mae = mean_absolute_error(train_data, predictions.fittedvalues)   
            params_evaluated.append(params)
            if (Error_MSE): results.append(mse)
            if (Error_MAE): results.append(mae)
        except:
            #print(f"Exception raised for {params}")
            params_evaluated.append(params)
            results.append(1e5)
        
        #print(params_evaluated, mse)
    return params_evaluated, results

def plot_net_demand(time_series,station_identifier,the_station_name):
    plt.figure(figsize=(10, 6))
    plt.plot(time_series.index, time_series , color='r', marker='s')  
    plt.title(f'Daily balance ride for Station: {the_station_name}')
    plt.xlabel('Date')
    plt.ylabel('Net Rides')
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=75)  
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FIGURE_DIR, f'Net_Distribution_{station_identifier}_{the_station_name}.png'))
    plt.show()
    

def forecast_and_evaluate_ARIMA(train_data, test_data, p, d, q, station_identifier,the_station_name, trend=None):
    # Fit the ARIMA model
    final_model = ARIMA(train_data, order=(p, d, q), trend=trend)
    fitted_final_model = final_model.fit()
    
    # Forecast the length of the test data
    forecast_steps = len(test_data)
    last_month_predictions = fitted_final_model.forecast(steps=forecast_steps)
    
    # Assign the index of test_data to the forecasted values
    last_month_predictions.index = test_data.index[:forecast_steps]
    
    # Create a DataFrame to hold both actual and predicted values for comparison
    comparison_df = pd.DataFrame({
        'Actual': test_data[:forecast_steps],
        'Predicted': last_month_predictions
    })
    
    print("Last Month Predictions vs. Actuals")
    print(comparison_df)
    
    # Calculate MAE and MSE using the Actual and Predicted columns from comparison_df
    mae_last_month = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
    mse_last_month = mean_squared_error(comparison_df['Actual'], comparison_df['Predicted'])
    
    # Print the results
    print(f'Mean Absolute Error (MAE) for the last month: {mae_last_month}')
    print(f'Mean Squared Error (MSE) for the last month: {mse_last_month}')
    
    # Visualize the comparison
    plt.figure(figsize=(10, 5))
    plt.plot(comparison_df.index, comparison_df['Actual'], label='Actual', color='blue', marker='o', markersize=8)
    plt.plot(comparison_df.index, comparison_df['Predicted'], label='Predicted', color='red', marker='s', markersize=8)
    plt.title(f'Actual vs Predicted Values for Last Month {the_station_name}')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.xticks(rotation=75)
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(os.path.join(FIGURE_DIR, f'Actual_vs_Predicted_Values_for_Last_Month_{station_identifier}_{the_station_name}.png'))
    plt.show()
    
    forecast_result = fitted_final_model.get_forecast(steps=forecast_steps)
    predicted_values = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

    # Plotting both the predicted values and confidence intervals
    plt.figure(figsize=(10, 5))
    plt.plot(test_data.index, test_data, label='Actual', color='blue', marker='o',markersize=8)
    plt.plot(test_data.index, predicted_values, label='Predicted', color='red', marker='s',markersize=8)
    plt.fill_between(test_data.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='lightgrey', alpha=0.5, label='95% Confidence Interval')
    plt.legend()
    plt.title(f'Actual vs Predicted with Confidence Intervals {the_station_name}')
    plt.xlabel('Days')
    plt.ylabel('Values')
    plt.xticks(rotation=75)
    plt.savefig(os.path.join(FIGURE_DIR, f'Actual_vs_Predicted_with_Confidence_Intervals_{station_identifier}_{the_station_name}.png'))
    plt.show()
    return mae_last_month, mse_last_month, comparison_df