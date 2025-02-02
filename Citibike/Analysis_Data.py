import os
import pandas as pd
import glob

def run_analysis(raw_data_directory, clean_data_directory):
    ###################### 
    ## 0. Loading 
    print("\n 0. Loading \n") 

    # Load and combine CSV files into one DataFrame
    file_pattern = os.path.join(raw_data_directory, "*.csv")
    csv_files = glob.glob(file_pattern)
    dfs = []
    i = 0
    # Print the file name
    for file in csv_files:
        i += 1
        print(f"{i} Loading file: {file}")  
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print("\nAll files loaded successfully.\n")

    ################################ 
    ## 1. Data Inspection 
    print("\n 1. Data Inspection \n") 

    print(df_combined.head())  
    print(f"Shape of the data: {df_combined.shape}")
    print(df_combined.info())  
    print(f"Number of duplicated rows: {df_combined.duplicated().sum()}")
      
    ################################ 
    ## 2. Handling Missing Data
    print("\n 2. Handling Missing Data\n")

    # Drop rows with missing values 
    df_combined = df_combined.dropna(subset=['started_at', 'ended_at'])
    df_combined['started_at'] = pd.to_datetime(df_combined['started_at'], errors='coerce')
    df_combined['ended_at'] = pd.to_datetime(df_combined['ended_at'], errors='coerce')
    df_combined = df_combined.dropna(subset=['started_at', 'ended_at'])  
    df_combined = df_combined.dropna()

    print("Rows with missing values have been removed.")

    ################################
    ## 3. Calculate ride duration in minutes
    print("\n 3. Calculate Ride Duration \n")
    df_combined['ride_duration'] = (df_combined['ended_at'] - df_combined['started_at']).dt.total_seconds() / 60
       
    # Add hour, day, and month for later aggregation
    df_combined['hour'] = df_combined['started_at'].dt.strftime('%Y-%m-%d %H:00:00')  
    df_combined['day'] = df_combined['started_at'].dt.date  
    df_combined['month'] = df_combined['started_at'].dt.to_period('M')

    ################################
    ## 4. Removing Outliers 
    print("\n 4. Removing Outliers\n")

    # Define the IQR method to remove outliers
    Q1 = df_combined['ride_duration'].quantile(0.25)
    Q3 = df_combined['ride_duration'].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    df_combined = df_combined[(df_combined['ride_duration'] >= lower_bound) & (df_combined['ride_duration'] <= upper_bound)]
    print(f"Outliers removed. Remaining rows: {df_combined.shape[0]}")

    ################################
    ## 5. Calculate Mean Location for Each Station
    print("\n 5. Calculating Mean Latitude and Longitude for Each Station\n")

    # Calculate the mean latitude and longitude for each start station
    start_station_location = df_combined.groupby('start_station_name').agg(
        mean_start_lat=('start_lat', 'mean'),
        mean_start_lng=('start_lng', 'mean')
    ).reset_index()

    # Calculate the mean latitude and longitude for each end station
    end_station_location = df_combined.groupby('end_station_name').agg(
        mean_end_lat=('end_lat', 'mean'),
        mean_end_lng=('end_lng', 'mean')
    ).reset_index()

    # Merge the mean latitude and longitude back into the original dataframe
    df_combined = df_combined.merge(start_station_location, on='start_station_name', how='left')
    df_combined = df_combined.merge(end_station_location, on='end_station_name', how='left')
    
    ################################
    ## 6. Data Aggregation 
    print("\n 6. Aggregating Data (Hourly, Daily, Monthly)\n")

    print("\n 6.1 Aggregating Data Hourly\n")
    # Aggregating data for the start stations 
    hourly_start_stats = df_combined.groupby(['start_station_name', 'start_station_id', 'mean_start_lat', 'mean_start_lng', 'hour']).agg(
        departure=('ride_id', 'count'),
        avg_duration_start=('ride_duration', 'mean')
    ).reset_index()
    # Aggregating data for the end stations 
    hourly_end_stats = df_combined.groupby(['end_station_name', 'end_station_id', 'mean_end_lat', 'mean_end_lng', 'hour']).agg(
        arrival=('ride_id', 'count'),
        avg_duration_end=('ride_duration', 'mean')
    ).reset_index()

    print("\n 6.2 Aggregating Data Daily\n")
    # Aggregating data for the start stations 
    daily_start_stats = df_combined.groupby(['start_station_name', 'start_station_id', 'mean_start_lat', 'mean_start_lng', 'day']).agg(
        departure=('ride_id', 'count'),
        avg_duration=('ride_duration', 'mean')
    ).reset_index()
    # Aggregating data for the end stations 
    daily_end_stats = df_combined.groupby(['end_station_name', 'end_station_id', 'mean_end_lat', 'mean_end_lng', 'day']).agg(
        arrival=('ride_id', 'count'),
        avg_duration=('ride_duration', 'mean')
    ).reset_index()
    
    print("\n 6.3 Aggregating Data Monthly\n")
    # Aggregate monthly start stats
    monthly_start_stats = df_combined.groupby(['start_station_name', 'start_station_id', 'mean_start_lat', 'mean_start_lng', 'month']).agg(
        departure=('ride_id', 'count'),
        avg_duration=('ride_duration', 'mean'),
        total_members=('member_casual', lambda x: (x == 'member').sum()),
        total_casual=('member_casual', lambda x: (x == 'casual').sum()),
        total_electric_bikes=('rideable_type', lambda x: (x == 'electric_bike').sum()),
        total_classic_bikes=('rideable_type', lambda x: (x == 'classic_bike').sum())
    ).reset_index()

    # Reorder the columns to include total_departures 
    monthly_start_stats = monthly_start_stats[['start_station_name', 'start_station_id', 'mean_start_lat', 'mean_start_lng', 'month', 
                                   'departure', 'avg_duration', 'total_members', 'total_casual', 
                                   'total_electric_bikes', 'total_classic_bikes']]
   
    monthly_end_stats = df_combined.groupby(['end_station_name', 'end_station_id', 'mean_end_lat', 'mean_end_lng', 'month']).agg(
        arrival=('ride_id', 'count'),
        avg_duration=('ride_duration', 'mean'),
    ).reset_index()

    # Reorder the columns to include total_arrival 
    monthly_end_stats = monthly_end_stats[['end_station_name', 'end_station_id', 'mean_end_lat', 'mean_end_lng', 'month', 
                                   'arrival', 'avg_duration']]

    # Clean monthly start and end station stats
    c_monthly_start_stats = clean_data(monthly_start_stats, ['start_station_name', 'start_station_id', 'month'])
    c_monthly_end_stats   = clean_data(monthly_end_stats, ['end_station_name', 'end_station_id', 'month'])

    # Clean daily start and end station stats
    c_daily_start_stats = clean_data(daily_start_stats, ['start_station_name', 'start_station_id', 'day'])
    c_daily_end_stats   = clean_data(daily_end_stats, ['end_station_name', 'end_station_id', 'day'])

    # Clean hourly start and end station stats
    c_hourly_start_stats = clean_data(hourly_start_stats, ['start_station_name', 'start_station_id', 'hour'])
    c_hourly_end_stats   = clean_data(hourly_end_stats, ['end_station_name', 'end_station_id', 'hour'])
    
    ################################
    ## 7. Saving the Aggregated Data 
    print("\n 7. Saving the Aggregated Data \n")

    # Save hourly aggregated data
    hourly_start_output_path = os.path.join(clean_data_directory, 'hourly_start_station_stats.csv')
    c_hourly_start_stats.to_csv(hourly_start_output_path, index=False , date_format='%Y-%m-%d %H:%M:%S')

    hourly_end_output_path = os.path.join(clean_data_directory, 'hourly_end_station_stats.csv')
    c_hourly_end_stats.to_csv(hourly_end_output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')

    # Save the daily aggregated data
    daily_start_output_path = os.path.join(clean_data_directory, 'daily_start_station_stats.csv')
    c_daily_start_stats.to_csv(daily_start_output_path, index=False)
    
    daily_end_output_path = os.path.join(clean_data_directory, 'daily_end_station_stats.csv')
    c_daily_end_stats.to_csv(daily_end_output_path, index=False)

    # Save monthly aggregated data
    monthly_start_output_path = os.path.join(clean_data_directory, 'monthly_start_station_stats.csv')
    c_monthly_start_stats.to_csv(monthly_start_output_path, index=False)

    monthly_end_output_path = os.path.join(clean_data_directory, 'monthly_end_station_stats.csv')
    c_monthly_end_stats.to_csv(monthly_end_output_path, index=False)

    ################################
    ### 8. Merge
    # Merge cleaned data and remove outlier
    merged_hourly_df = merge_and_process_station_stats(c_hourly_start_stats, c_hourly_end_stats, 'hour')
    m_hourly_start_output_path = os.path.join(clean_data_directory, 'Merge_hourly_station_stats.csv')
    merged_hourly_df.to_csv(m_hourly_start_output_path, index=False)
    print("Hourly merged data saved to merge_hourly_station_stats.csv")
    
    merged_daily_df = merge_and_process_station_stats(c_daily_start_stats, c_daily_end_stats, 'day')
    m_daily_start_output_path = os.path.join(clean_data_directory, 'Merge_daily_station_stats.csv')
    merged_daily_df.to_csv(m_daily_start_output_path, index=False)
    print("Daily merged data saved to merge_daily_station_stats.csv")

    merged_monthly_df = merge_and_process_station_stats(c_monthly_start_stats, c_monthly_end_stats, 'month')
    m_monthly_start_output_path = os.path.join(clean_data_directory, 'Merge_monthly_station_stats.csv')
    merged_monthly_df.to_csv(m_monthly_start_output_path, index=False)
    print("Monthly merged data saved to merge_monthly_station_stats.csv")

# Function to clean data
def clean_data(data, subset_columns):
    duplicates = data.duplicated(subset=subset_columns, keep=False)

    if duplicates.any():
        print(f"Duplicate rows found :")
        print(data[duplicates])
        data_I = data.drop_duplicates(subset=subset_columns)
        print("Duplicate rows have been removed.")
    else:
        print(f"No duplicate rows found.")
        data_I = data
        
    return data_I

def merge_and_process_station_stats(start_df, end_df, time_unit):
     # Rename the end station name and id columns for clarity during merging
    end_df.rename(columns={
        'end_station_name': 'start_station_name',
        'end_station_id': 'start_station_id'
    }, inplace=True)

    # Merge DataFrames
    merged_df = pd.merge(start_df, end_df[['start_station_name', 'start_station_id', 'arrival', time_unit]],
                         on=['start_station_name', 'start_station_id', time_unit],
                         how='left')

    # Replace NaN values in 'arrival' with 0
    merged_df['arrival'] = merged_df['arrival'].fillna(0)

    # Calculate net_ride as the difference between arrival and departure
    merged_df['net_ride'] = merged_df['arrival'] - merged_df['departure']

    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Remove outliers for the net_ride column
    final_df = remove_outliers_iqr(merged_df, 'net_ride')

    # Sort the final DataFrame by station name, station ID, and then by date
    final_df.sort_values(by=['start_station_name', 'start_station_id', time_unit], inplace=True)
    
    return final_df

# Create a new function to save only specific columns
def save_station_info(start_station_df,clean_data_directory):
    # Select only the required columns
    station_info = start_station_df[['start_station_id', 'start_station_name', 'mean_start_lat', 'mean_start_lng']]
    station_info = station_info.drop_duplicates()
    station_info.to_csv(os.path.join(clean_data_directory,'station_info.csv'), index=False)
    print("Station information saved to 'station_info.csv'")