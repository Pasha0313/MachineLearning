import os
import pandas as pd
import Analysis_Data as A_Data
import Visualizations as viz  
import Forcasting as Forc_F  
import Analysis_Pro as A_Post

# Set the directories
raw_data_directory = r"C:\DATA\Job\Causalens\DATA\2023-citibike-tripdata"
clean_data_directory = r"C:\DATA\Job\Causalens\Aggregated_Data"
output_dir = r"C:\DATA\Job\Causalens\Figures"
os.makedirs(output_dir, exist_ok=True)
date_format = "%Y-%m-%d %H:00:00" 

cleaning_aggregate = False
Optim_data         = False
classifing_data    = False
Forcasting_data    = True
Arima_Tuner        = False
station_id         = 6450.12  # Replace with actual station ID

Load_Hourly  = False
Load_daily   = True
Load_Monthly = True

Plot_Hourly  = False
Plot_daily   = False
Plot_Monthly = False

# Run the full analysis pipeline
if cleaning_aggregate:
    A_Data.run_analysis(raw_data_directory, clean_data_directory)
    print("Data cleaning and aggregation completed.")

# File paths for your datasets
if (Load_Hourly) :
    print("Hourly Data loading ...")
    Hourly_start_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\Hourly_start_station_stats.csv'    
    Hourly_end_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\Hourly_end_station_stats.csv'
    Hourly_combined_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\Merge_hourly_station_stats.csv'

    Hourly_start_station_stats = pd.read_csv(
        Hourly_start_station_stats_path,
        dtype={'start_station_name': 'str', 'start_station_id': 'str', 'departure': 'int64', 'avg_duration_start': 'float64'},low_memory=False)
    Hourly_end_station_stats = pd.read_csv(
        Hourly_end_station_stats_path,
        dtype={'end_station_name': 'str', 'end_station_id': 'str', 'arrival': 'int64', 'avg_duration_end': 'float64'},low_memory=False)
    Hourly_combined_station_stats = pd.read_csv(
        Hourly_combined_station_stats_path,
        dtype={'start_station_name': 'str', 'start_station_id': 'str', 'mean_start_lat':'float64', 'mean_start_lng': 'float64', 'departure': 'int64', 'avg_duration_end': 'float64', 'arrival' : 'float64', 'net_ride' : 'float64'},low_memory=False)

    Hourly_start_station_stats['hour'] = pd.to_datetime(Hourly_start_station_stats['hour'], format='%Y-%m-%d %H:00:00')
    Hourly_end_station_stats['hour'] = pd.to_datetime(Hourly_end_station_stats['hour'], format='%Y-%m-%d %H:00:00')
    Hourly_combined_station_stats['hour'] = pd.to_datetime(Hourly_combined_station_stats['hour'], format='%Y-%m-%d %H:00:00')
    print("Hourly station stats loaded.")

if (Load_daily):
    print("Daily Data loading ...")
    daily_start_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\daily_start_station_stats.csv'
    daily_end_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\daily_end_station_stats.csv'
    daily_combined_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\Merge_daily_station_stats.csv'

    
    daily_start_station_stats = pd.read_csv(daily_start_station_stats_path, parse_dates=['day'],low_memory=False)
    daily_end_station_stats = pd.read_csv(daily_end_station_stats_path, parse_dates=['day'],low_memory=False)
    daily_combined_station_stats = pd.read_csv(daily_combined_station_stats_path, parse_dates=['day'],low_memory=False)
    print("Daily station stats loaded.")

if (Load_Monthly):
    print("Monthly Data loading ...")    
    monthly_start_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\monthly_start_station_stats.csv'
    monthly_end_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\monthly_end_station_stats.csv'
    monthly_combined_station_stats_path = r'C:\DATA\Job\Causalens\Aggregated_Data\Merge_monthly_station_stats.csv'    

    monthly_start_station_stats = pd.read_csv(monthly_start_station_stats_path, parse_dates=['month'],low_memory=False)
    monthly_end_station_stats = pd.read_csv(monthly_end_station_stats_path, parse_dates=['month'],low_memory=False)
    monthly_combined_station_stats = pd.read_csv(monthly_combined_station_stats_path, parse_dates=['month'],low_memory=False)
    print("Monthly station stats loaded.")

# Call visualization functions for Hourly stats
if (Load_Hourly and Plot_Hourly):
    print("Generating Hourly plots...")
    viz.plot_total_rides_by_hour(Hourly_start_station_stats)
    viz.plot_average_rides_by_hour(Hourly_start_station_stats)
    print("Hourly plots saved.")

# Call visualization functions for daily stats
if (Load_daily and Plot_daily):
    print("Generating daily plots...")
    viz.plot_avg_daily_rides_by_top_stations(daily_start_station_stats,daily_end_station_stats,top_n=10)
    viz.plot_avg_duration_by_top_stations(daily_start_station_stats,top_n=10)
    viz.plot_rides_over_time_daily(daily_start_station_stats)
    viz.plot_top_stations_daily_demand(daily_combined_station_stats,top_n=3)
    viz.plot_average_rides_by_day_of_week(daily_start_station_stats)
    viz.plot_total_rides_by_day_of_week(daily_start_station_stats)
    print("Daily plots saved.")

# Call visualization functions for monthly stats
if (Load_Monthly and Plot_Monthly):
    print("Generating monthly plots...")
    viz.plot_total_rides_by_top_station_monthly(monthly_start_station_stats,monthly_end_station_stats, top_n=10)
    viz.plot_avg_duration_by_top_station_monthly(monthly_start_station_stats, top_n=10)
    viz.plot_rides_by_bike_type_monthly_I(monthly_start_station_stats)
    viz.plot_rides_by_member_type_monthly_I(monthly_start_station_stats)
    viz.plot_rides_by_bike_type_monthly_II(monthly_start_station_stats)
    viz.plot_rides_by_member_type_monthly_II(monthly_start_station_stats)
    print("Monthly plots saved.")

    # Generate and save heatmap for monthly stats
    print("Generating monthly heatmap...")
    viz.plot_station_heatmap_monthly(monthly_start_station_stats,monthly_end_station_stats)
    print("Monthly heatmap saved.")

if (Optim_data):
    print("Daily analysis start.")
    A_Post.analyze_departures_arrival_stats(daily_start_station_stats, daily_end_station_stats,'day',top_n=30,screen=True)
    print("Daily analysis finish.")
    
    print("Monthly analysis start.")
    A_Post.analyze_departures_arrival_stats(monthly_start_station_stats, monthly_end_station_stats,'month',top_n=30,screen=True)
    print("Monthly analysis finish.")

if (classifing_data):
    viz.plot_classified_heatmap(monthly_start_station_stats,monthly_end_station_stats)

if (Forcasting_data) :
    Forc_F.Forcast_daily_distribution(daily_start_station_stats, daily_end_station_stats, station_id,Arima_Tuner, by_id=True) 

  