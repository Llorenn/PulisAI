"""
Data Processing Pipeline for Crime Data Upload and Model Retraining
This module handles the complete pipeline from raw CSV files to model-ready aggregated data.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


# Police Station Coordinates in Angeles City
POLICE_STATIONS = {
    "Station": ["Police Station 1", "Police Station 2", "Police Station 3",
                "Police Station 4", "Police Station 5", "Police Station 6"],
    "Latitude": [15.1350, 15.1614, 15.1607, 15.1600, 15.1446, 15.15884],
    "Longitude": [120.5910, 120.6087, 120.6089, 120.5918, 120.5572, 120.59226]
}

# Focus Crime Categories
FOCUS_CRIMES = ['Murder', 'Robbery', 'Rape', 'Physical Injuries', 'Homicide', 'Theft', 'Carnapping MC', 'Carnapping MV']

# Crime Pattern Definitions
CRIME_PATTERNS = {
    'murder': "MURDER|PARRICIDE",
    'homicide': "^HOMICIDE|ATTEMPTED HOMICIDE|FRUSTRATED HOMICIDE",
    'rape': "RAPE|ANTI-RAPE LAW|11648|RAPE WITH",
    'physical_injuries': "PHYSICAL INJURIES|SERIOUS PHYSICAL INJURIES|LESS SERIOUS PHYSICAL INJURIES|SLIGHT PHYSICAL INJURIES",
    'robbery': "ROBBERY|^ROBBERY WITH",
    'theft': "THEFT|QUALIFIED THEFT",
    'carnapping_mc': "CARNAPPING MC",
    'carnapping_mv': "CARNAPPING MV"
}


def process_crimes_data(crimes_df):
    """
    Process crimes dataset: add temporal features, time bins, focus crime classification,
    police station distances, and clean missing values.

    Args:
        crimes_df: Raw crimes DataFrame

    Returns:
        Processed crimes DataFrame
    """
    print("Processing crimes data...")

    # Make a copy to avoid modifying original
    crimes = crimes_df.copy()

    # Convert Date to datetime
    crimes['Date'] = pd.to_datetime(crimes['Date'])

    # Ensure Year column is string type (for consistency with filter options)
    if 'Year' in crimes.columns:
        crimes['Year'] = crimes['Year'].astype(str)

    # Extract temporal features
    crimes['Weekday'] = crimes['Date'].dt.isocalendar().day
    crimes['Day_of_Week'] = crimes['Date'].dt.day_name()
    crimes['Week_of_Year'] = crimes['Date'].dt.isocalendar().week
    crimes['Quarter'] = crimes['Date'].dt.quarter
    crimes['Is_Weekend'] = crimes['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

    # Create Time_of_Day bins (Midnight, Morning, Afternoon, Evening)
    hour_bins = [0, 6, 12, 18, 24]
    hour_labels = ['Midnight', 'Morning', 'Afternoon', 'Evening']
    crimes['Time_of_Day'] = pd.cut(crimes['Hour'], bins=hour_bins, labels=hour_labels, right=False)

    # Classify Focus Crimes
    conditions = [
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['murder'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['robbery'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['rape'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['physical_injuries'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['homicide'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['theft'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['carnapping_mc'], case=False, na=False),
        crimes['Offense Committed'].str.contains(CRIME_PATTERNS['carnapping_mv'], case=False, na=False)
    ]
    crimes['Focus_Crime'] = np.select(conditions, FOCUS_CRIMES, default='Other')

    # Handle missing values in 'Time Committed'
    crimes['Time Committed'] = crimes.groupby(['Barangay', 'Offense Committed'])['Time Committed'].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    if crimes['Time Committed'].isna().sum() > 0:
        crimes['Time Committed'] = crimes['Time Committed'].fillna(
            crimes.groupby('Barangay')['Time Committed'].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
        )

    # Handle missing values in Latitude/Longitude
    crimes['Latitude'] = crimes['Latitude'].fillna(
        crimes.groupby(['Barangay', 'Offense Committed'])['Latitude'].transform('mean')
    )
    if crimes['Latitude'].isna().sum() > 0:
        crimes['Latitude'] = crimes['Latitude'].fillna(crimes.groupby('Barangay')['Latitude'].transform('mean'))

    crimes['Longitude'] = crimes['Longitude'].fillna(
        crimes.groupby(['Barangay', 'Offense Committed'])['Longitude'].transform('mean')
    )
    if crimes['Longitude'].isna().sum() > 0:
        crimes['Longitude'] = crimes['Longitude'].fillna(crimes.groupby('Barangay')['Longitude'].transform('mean'))

    # Clean Offense ID
    crimes['Offense ID'] = crimes['Offense ID'].astype(str).str.replace(" ", "")

    # Drop 'Case Solved Type' if exists
    if 'Case Solved Type' in crimes.columns:
        crimes.drop(columns=['Case Solved Type'], inplace=True)

    # Add police station features
    crimes = add_police_station_features(crimes)

    # Reorder columns
    column_order = ['Offense ID', 'Barangay', 'Date', 'Time Committed', 'Offense Committed',
                    'Focus_Crime', 'Crime Type', 'Case Status', 'Latitude', 'Longitude',
                    'Victim Count', 'Suspect Count', 'Year', 'Month', 'Day', 'Day_of_Week',
                    'Hour', 'Weekday', 'Week_of_Year', 'Quarter', 'Is_Weekend', 'Time_of_Day',
                    'Police Station', 'Distance_from_Police', 'Nearest_Police_Station',
                    'Nearest_Police_Distance', 'Num_Police_Stations_1km']

    # Only keep columns that exist
    existing_columns = [col for col in column_order if col in crimes.columns]
    crimes = crimes[existing_columns]

    print(f"   Crimes data processed: {crimes.shape}")
    return crimes


def add_police_station_features(crimes):
    """Add police station distance and proximity features."""

    # Create station coordinates dictionary
    station_coords = {}
    for i in range(len(POLICE_STATIONS['Station'])):
        station_coords[POLICE_STATIONS['Station'][i]] = (
            POLICE_STATIONS['Latitude'][i],
            POLICE_STATIONS['Longitude'][i]
        )

    # Calculate distance to assigned police station
    def get_distance_to_station(row, assigned_station):
        if assigned_station in station_coords:
            crime_coords = (row['Latitude'], row['Longitude'])
            station_coord = station_coords[assigned_station]
            return geodesic(crime_coords, station_coord).meters / 1000  # km
        return None

    crimes['Distance_from_Police'] = crimes.apply(
        lambda row: get_distance_to_station(row, row['Police Station']), axis=1
    )

    # Find nearest station and its distance
    def get_nearest_station(row):
        crime_coords = (row['Latitude'], row['Longitude'])
        nearest_station = None
        min_distance = float('inf')

        for station, coords in station_coords.items():
            distance = geodesic(crime_coords, coords).meters / 1000
            if distance < min_distance:
                min_distance = distance
                nearest_station = station

        return pd.Series([nearest_station, min_distance])

    crimes[['Nearest_Police_Station', 'Nearest_Police_Distance']] = crimes.apply(get_nearest_station, axis=1)

    # Count police stations within 1km
    def count_nearby_stations(row, radius_km=1):
        crime_coords = (row['Latitude'], row['Longitude'])
        count = 0
        for coords in station_coords.values():
            distance = geodesic(crime_coords, coords).meters / 1000
            if distance <= radius_km:
                count += 1
        return count

    crimes['Num_Police_Stations_1km'] = crimes.apply(count_nearby_stations, axis=1)

    return crimes


def process_suspects_data(suspects_df):
    """
    Process suspects dataset: clean ages, handle missing values, create age groups.

    Args:
        suspects_df: Raw suspects DataFrame

    Returns:
        Processed suspects DataFrame
    """
    print("Processing suspects data...")

    suspects = suspects_df.copy()

    # Replace age 0 with NaN
    suspects['Age'] = suspects['Age'].replace(0, np.nan)

    # Fill missing ages with grouped mean
    suspects['Age'] = suspects['Age'].fillna(
        suspects.groupby(['Barangay', 'Offense (Consolidated)'])['Age'].transform('mean')
    )
    if suspects['Age'].isna().sum() > 0:
        suspects['Age'] = suspects['Age'].fillna(suspects.groupby('Barangay')['Age'].transform('mean'))

    # Fill missing gender with grouped mode
    suspects['Gender'] = suspects['Gender'].fillna(
        suspects.groupby(['Barangay', 'Offense (Consolidated)'])['Gender'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        )
    )

    # Drop columns with many missing values
    suspects.drop(columns=['Nationality', 'Civil Status'], inplace=True, errors='ignore')

    # Create age groups
    age_bins = [0, 17, 25, 34, 44, 54, 64, np.inf]
    age_labels = ['0-17', '18-25', '26-34', '35-44', '45-54', '55-64', '65+']
    suspects['Age_Group'] = pd.cut(suspects['Age'], bins=age_bins, labels=age_labels, right=True)

    # Clean Offense ID
    suspects['Offense ID'] = suspects['Offense ID'].astype(str).str.replace(" ", "")

    print(f"   Suspects data processed: {suspects.shape}")
    return suspects


def process_victims_data(victims_df):
    """
    Process victims dataset: clean ages, handle missing values, create age groups.

    Args:
        victims_df: Raw victims DataFrame

    Returns:
        Processed victims DataFrame
    """
    print("Processing victims data...")

    victims = victims_df.copy()

    # Replace age 0 with NaN
    victims['Age'] = victims['Age'].replace(0, np.nan)

    # Fill missing ages with grouped mean
    victims['Age'] = victims['Age'].fillna(
        victims.groupby(['Barangay', 'Offense (Consolidated)'])['Age'].transform('mean')
    )
    if victims['Age'].isna().sum() > 0:
        victims['Age'] = victims['Age'].fillna(victims.groupby('Barangay')['Age'].transform('mean'))

    # Fill missing gender with grouped mode
    victims['Gender'] = victims['Gender'].fillna(
        victims.groupby(['Barangay', 'Offense (Consolidated)'])['Gender'].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        )
    )

    # Drop columns with many missing values
    victims.drop(columns=['Nationality', 'Civil Status'], inplace=True, errors='ignore')

    # Create age groups
    age_bins = [0, 17, 25, 34, 44, 54, 64, np.inf]
    age_labels = ['0-17', '18-25', '26-34', '35-44', '45-54', '55-64', '65+']
    victims['Age_Group'] = pd.cut(victims['Age'], bins=age_bins, labels=age_labels, right=True)

    # Clean Offense ID
    victims['Offense ID'] = victims['Offense ID'].astype(str).str.replace(" ", "")

    print(f"   Victims data processed: {victims.shape}")
    return victims


def aggregate_suspects(suspects_df):
    """Aggregate suspects data by Offense ID."""
    agg_suspects = suspects_df.groupby('Offense ID').agg(
        Num_Suspects=('Age', 'count'),
        Avg_Suspects_Age=('Age', 'mean'),
        Male_Suspects=('Gender', lambda x: (x == 'Male').sum()),
        Female_Suspects=('Gender', lambda x: (x == 'Female').sum()),
        Suspects_0_17=('Age_Group', lambda x: (x == '0-17').sum()),
        Suspects_18_25=('Age_Group', lambda x: (x == '18-25').sum()),
        Suspects_26_34=('Age_Group', lambda x: (x == '26-34').sum()),
        Suspects_35_44=('Age_Group', lambda x: (x == '35-44').sum()),
        Suspects_45_54=('Age_Group', lambda x: (x == '45-54').sum()),
        Suspects_55_64=('Age_Group', lambda x: (x == '55-64').sum()),
        Suspects_65_Above=('Age_Group', lambda x: (x == '65+').sum())
    ).reset_index()

    return agg_suspects


def aggregate_victims(victims_df):
    """Aggregate victims data by Offense ID."""
    agg_victims = victims_df.groupby('Offense ID').agg(
        Num_Victims=('Age', 'count'),
        Avg_Victims_Age=('Age', 'mean'),
        Male_Victims=('Gender', lambda x: (x == 'Male').sum()),
        Female_Victims=('Gender', lambda x: (x == 'Female').sum()),
        Victims_0_17=('Age_Group', lambda x: (x == '0-17').sum()),
        Victims_18_25=('Age_Group', lambda x: (x == '18-25').sum()),
        Victims_26_34=('Age_Group', lambda x: (x == '26-34').sum()),
        Victims_35_44=('Age_Group', lambda x: (x == '35-44').sum()),
        Victims_45_54=('Age_Group', lambda x: (x == '45-54').sum()),
        Victims_55_64=('Age_Group', lambda x: (x == '55-64').sum()),
        Victims_65_Above=('Age_Group', lambda x: (x == '65+').sum())
    ).reset_index()

    return agg_victims


def merge_datasets(crimes, suspects, victims, barangay_info):
    """
    Merge all datasets to create enriched_df.

    Args:
        crimes: Processed crimes DataFrame
        suspects: Processed suspects DataFrame
        victims: Processed victims DataFrame
        barangay_info: Barangay demographic information DataFrame

    Returns:
        enriched_df: Merged and enriched DataFrame
    """
    print("Merging datasets...")

    # Aggregate suspects and victims by Offense ID
    agg_suspects = aggregate_suspects(suspects)
    agg_victims = aggregate_victims(victims)

    # Merge crimes with suspects (left join)
    enriched_df = crimes.merge(agg_suspects, on='Offense ID', how='left')

    # Merge with victims (left join)
    enriched_df = enriched_df.merge(agg_victims, on='Offense ID', how='left')

    # Merge with barangay info (left join on Barangay)
    enriched_df = enriched_df.merge(barangay_info, on='Barangay', how='left')

    # Fill NaN values for suspect/victim counts with 0
    suspect_victim_cols = [col for col in enriched_df.columns if
                          ('Num_Suspects' in col or 'Num_Victims' in col or
                           'Male_' in col or 'Female_' in col or
                           'Suspects_' in col or 'Victims_' in col or
                           'Avg_Suspects' in col or 'Avg_Victims' in col)]

    for col in suspect_victim_cols:
        enriched_df[col] = enriched_df[col].fillna(0)

    print(f"   Datasets merged: {enriched_df.shape}")
    return enriched_df


def create_focus_df(enriched_df):
    """
    Filter enriched_df to only include focus crimes.

    Args:
        enriched_df: Merged and enriched DataFrame

    Returns:
        focus_df: DataFrame with only focus crimes
    """
    print("Creating focus_df...")

    focus_df = enriched_df[enriched_df['Focus_Crime'] != 'Other'].copy()

    print(f"   Focus crimes filtered: {focus_df.shape}")
    return focus_df


def aggregate_for_training(df):
    """
    Aggregate focus_df by Barangay, Month, Weekday, Time_of_Day for model training.
    This is the same aggregate function used in the original model training.

    Args:
        df: focus_df DataFrame

    Returns:
        Aggregated DataFrame ready for training
    """
    print("Aggregating data for training...")

    # Clean rows with missing grouping columns first
    df_clean = df.dropna(subset=['Barangay', 'Month', 'Weekday', 'Time_of_Day'])

    agg_df = df_clean.groupby(['Barangay', 'Month', 'Weekday', 'Time_of_Day']).agg(
        Crime_Count=('Offense ID', 'count'),

        # Temporal features
        Avg_Hour=('Hour', 'mean'),
        Mode_Hour=('Hour', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
        Weekend_Crimes=('Is_Weekend', 'sum'),
        Weekday_Crimes=('Is_Weekend', lambda x: (~x.astype(bool)).sum()),

        # Spatial/demographic
        Population=('Population_2024', 'first'),
        Pop_Density=('Pop_Density_2024', 'first'),
        Area_sqkm=('Area_sqkm', 'first'),

        # Police presence
        Avg_Distance_Police=('Distance_from_Police', 'mean'),
        Avg_Num_Stations_1km=('Num_Police_Stations_1km', 'mean'),

        # Crime characteristics
        Avg_Victims=('Num_Victims', 'median'),
        Avg_Suspects=('Num_Suspects', 'median'),

        # Most common crime type
        Mode_Focus_Crime=('Focus_Crime', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),

        # Focus crime distribution
        Murder_Count=('Focus_Crime', lambda x: (x == 'Murder').sum()),
        Theft_Count=('Focus_Crime', lambda x: (x == 'Theft').sum()),
        Robbery_Count=('Focus_Crime', lambda x: (x == 'Robbery').sum()),
        Physical_Injuries_Count=('Focus_Crime', lambda x: (x == 'Physical Injuries').sum()),
        Rape_Count=('Focus_Crime', lambda x: (x == 'Rape').sum()),
        Homicide_Count=('Focus_Crime', lambda x: (x == 'Homicide').sum()),
        Carnapping_MC_Count=('Focus_Crime', lambda x: (x == 'Carnapping MC').sum()),
        Carnapping_MV_Count=('Focus_Crime', lambda x: (x == 'Carnapping MV').sum())
    ).reset_index()

    # Calculate derived features
    agg_df['Crime_Rate_per_1000'] = (agg_df['Crime_Count'] / agg_df['Population']) * 1000
    agg_df['Crime_Density_sqkm'] = agg_df['Crime_Count'] / agg_df['Area_sqkm']
    agg_df['Weekend_Ratio'] = agg_df['Weekend_Crimes'] / (agg_df['Crime_Count'] + 1e-6)

    # Sort by grouping columns
    agg_df = agg_df.sort_values(['Barangay', 'Month', 'Weekday', 'Time_of_Day'])

    print(f"   Data aggregated: {agg_df.shape}")
    return agg_df


def classify_alarm_levels(new_df, baseline_df=None):
    """
    Classify crime alarm levels based on Crime_Count quantiles.

    Args:
        new_df: New aggregated DataFrame to classify
        baseline_df: Baseline DataFrame (2017-2024) to get thresholds from.
                    If None, use new_df itself.

    Returns:
        new_df with Alarm_Level column added
        q25, q75 thresholds
    """
    print("Classifying alarm levels...")

    # Use baseline data to determine thresholds, or use new_df if no baseline
    threshold_df = baseline_df if baseline_df is not None else new_df

    q25 = threshold_df['Crime_Count'].quantile(0.25)
    q75 = threshold_df['Crime_Count'].quantile(0.75)
    mean_crime = threshold_df['Crime_Count'].mean()

    print(f"ALARM THRESHOLDS:")
    print(f"   25th Percentile (Low/Medium boundary): {q25:.2f}")
    print(f"   75th Percentile (Medium/High boundary): {q75:.2f}")
    print(f"   Mean Crime Count: {mean_crime:.2f}")

    def classify(count):
        if count <= q25:
            return 'Low'
        elif count <= q75:
            return 'Medium'
        else:
            return 'High'

    new_df['Alarm_Level'] = new_df['Crime_Count'].apply(classify)

    print(f"\nAlarm Level Distribution:")
    print(new_df['Alarm_Level'].value_counts().sort_index())

    return new_df, q25, q75


def process_uploaded_data(crime_file, suspect_file, victim_file, barangay_info_file):
    """
    Complete pipeline to process uploaded raw data files.

    Args:
        crime_file: Path or DataFrame of crimes CSV
        suspect_file: Path or DataFrame of suspects CSV
        victim_file: Path or DataFrame of victims CSV
        barangay_info_file: Path or DataFrame of barangay info CSV

    Returns:
        focus_df: Processed and merged focus crimes DataFrame
    """
    print("\n" + "="*60)
    print("STARTING DATA PROCESSING PIPELINE")
    print("="*60 + "\n")

    # Load data if paths are provided
    if isinstance(crime_file, str):
        crimes = pd.read_csv(crime_file, parse_dates=['Date'])
    else:
        crimes = crime_file.copy()
        crimes['Date'] = pd.to_datetime(crimes['Date'])

    if isinstance(suspect_file, str):
        suspects = pd.read_csv(suspect_file)
    else:
        suspects = suspect_file.copy()

    if isinstance(victim_file, str):
        victims = pd.read_csv(victim_file)
    else:
        victims = victim_file.copy()

    if isinstance(barangay_info_file, str):
        barangay_info = pd.read_csv(barangay_info_file)
    else:
        barangay_info = barangay_info_file.copy()

    # Process each dataset
    crimes_processed = process_crimes_data(crimes)
    suspects_processed = process_suspects_data(suspects)
    victims_processed = process_victims_data(victims)

    # Merge datasets
    enriched_df = merge_datasets(crimes_processed, suspects_processed, victims_processed, barangay_info)

    # Create focus_df
    focus_df = create_focus_df(enriched_df)

    print("\n" + "="*60)
    print("DATA PROCESSING PIPELINE COMPLETED")
    print("="*60 + "\n")

    return focus_df
