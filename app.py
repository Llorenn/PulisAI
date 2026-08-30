from flask import Flask, render_template, request, url_for, redirect, session, flash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import json
from functools import wraps
import os
import secrets
from werkzeug.utils import secure_filename
import data_processor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import calendar

app = Flask(__name__)

# Signs the session cookie. Generated per process when unset, which logs
# everyone out on restart -- set PULISAI_SECRET_KEY to keep sessions alive.
app.secret_key = os.environ.get('PULISAI_SECRET_KEY') or secrets.token_hex(32)

# LOGIN CREDENTIALS
# Read from the environment so no working password sits in the repository.
# The default exists only so a fresh clone runs; override both before any
# deployment:  export PULISAI_USER=... PULISAI_PASSWORD=...
VALID_USERS = {
    os.environ.get('PULISAI_USER', 'admin'):
        os.environ.get('PULISAI_PASSWORD', 'pulisai'),
}

#LOGIN DECORATOR
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Barangay Name Mapping (Crime Dataset -> GeoJSON)
BARANGAY_NAME_MAPPING = {
    'AGAPITO DEL ROSARIO': 'Agapito del Rosario',
    'ANUNAS': 'Anunas',
    'BALIBAGO': 'Balibago',
    'CAPAYA': 'Capaya',
    'CLARO M RECTO': 'Claro M. Recto',
    'CUAYAN': 'Cuayan',
    'CUTCUT': 'Cutcut',
    'CUTUD': 'Cutud',
    'LOURDES NORTHWEST': 'Lourdes North West',
    'LOURDES SUR': 'Lourdes Sur',
    'LOURDES SUR EAST': 'Lourdes Sur East',
    'MALABANIAS': 'Malabanias',
    'MARGOT': 'Margot',
    'MINING': 'Mining',
    'PAMPANG': 'Pampang',
    'PANDAN': 'Pandan',
    'PULUNG MARAGUL': 'Pulung Maragul',
    'PULUNGBULO': 'Pulungbulu',
    'PULUNG CACUTUD': 'Pulung Cacutud',
    'SALAPUNGAN': 'Salapungan',
    'SAN JOSE': 'San Jose',
    'SAN NICOLAS': 'San Nicolas',
    'STA TERESITA': 'Santa Teresita',
    'STA TRINIDAD': 'Santa Trinidad',
    'STO CRISTO': 'Santo Cristo',
    'STO DOMINGO': 'Santo Domingo',
    'STO ROSARIO': 'Santo Rosario (Pob.)',
    'SAPALIBUTAD': 'Sapalibutad',
    'SAPANGBATO': 'Sapangbato',
    'TABUN': 'Tabun',
    'VIRGEN DELOS REMEDIOS': 'Virgen Delos Remedios',
    'AMSIC': 'Amsic',
    'NINOY AQUINO': 'Ninoy Aquino (Marisol)'
}

#Data Loading and Column Definitions (from Gradio app)

COL_DATASET = "data/focus_df.csv"
COL_CRIME_DESC = "Focus_Crime"
COL_LAT = "Latitude"
COL_LON = "Longitude"
COL_DATE = "Date"
COL_STATUS = "Case Status"
COL_AGE = "Avg_Victims_Age"
COL_AREA = "Barangay"
COL_MALE_VICTIMS = "Male_Victims"
COL_FEMALE_VICTIMS = "Female_Victims"
COL_YEAR = "Year"
COL_TIME_OF_DAY = "Time_of_Day"
COL_DAY_OF_WEEK = "Day_of_Week"
COL_OFFENSE = "Offense Committed"

#Month Names Mapping (used in both visualization and prediction)
MONTH_NAMES = {
    1: 'January (Q1)', 2: 'February (Q1)', 3: 'March (Q1)', 4: 'April (Q2)',
    5: 'May (Q2)', 6: 'June (Q2)', 7: 'July (Q3)', 8: 'August (Q3)',
    9: 'September (Q3)', 10: 'October (Q4)', 11: 'November (Q4)', 12: 'December (Q4)'
}

# Reverse mapping for form submission
MONTH_NAME_TO_NUMBER = {v: k for k, v in MONTH_NAMES.items()}

VICTIM_AGE_COLS = [
    'Victims_0_17', 'Victims_18_25', 'Victims_26_34', 'Victims_35_44',
    'Victims_45_54', 'Victims_55_64', 'Victims_65_Above'
]

COL_OFFENSE_ID = 'Offense ID'
COL_HOUR = 'Hour'
COL_IS_WEEKEND = 'Is_Weekend'
COL_POPULATION = 'Population_2024'
COL_POP_DENSITY = 'Pop_Density_2024'
COL_AREA_SQKM = 'Area_sqkm'
COL_DISTANCE_POLICE = 'Distance_from_Police'
COL_NUM_STATIONS_1KM = 'Num_Police_Stations_1km'
COL_NUM_VICTIMS = 'Num_Victims'
COL_NUM_SUSPECTS = 'Num_Suspects'
COL_FOCUS_CRIME = 'Focus_Crime'


# Load dataset FOR PLOTTING
try:
    df = pd.read_csv(COL_DATASET)
    for col in VICTIM_AGE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if COL_YEAR in df.columns:
         df[COL_YEAR] = df[COL_YEAR].astype(str)
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce', dayfirst=True).dt.strftime('%Y-%m-%d')

except FileNotFoundError:
    print(f"Error: The dataset file '{COL_DATASET}' was not found.")
    df = pd.DataFrame()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    df = pd.DataFrame()

#Dynamic Options (from DataFrame) FOR PLOTTING
if not df.empty:
    sorted_crimes = sorted(df[COL_CRIME_DESC].unique())
    CRIME_OPTIONS = np.insert(sorted_crimes, 0, "ALL")

    sorted_barangays = sorted(df[COL_AREA].unique())
    BARANGAY_OPTIONS = np.insert(sorted_barangays, 0, "ALL")

    sorted_years = sorted(df[COL_YEAR].unique(), reverse=True)
    YEAR_OPTIONS = np.insert(sorted_years, 0, "ALL")

    # New filter options for Visualizations page
    MONTH_OPTIONS_VIZ = ["ALL"] + [MONTH_NAMES[i] for i in range(1, 13)]
    WEEKDAY_OPTIONS_VIZ = ["ALL", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    TIME_OF_DAY_OPTIONS_VIZ = ["ALL", "Morning", "Afternoon", "Evening", "Midnight"]
else:
    # Fallback options
    CRIME_OPTIONS = ["ALL"] + sorted(["CARNAPPING MC", "CARNAPPING MV", "HOMICIDE", "MURDER", "RAPE", "ROBBERY", "PHYSICAL INJURIES", "THEFT"])
    BARANGAY_OPTIONS = ["ALL"] + sorted(["PAMPANGA", "MALABANIAS", "BALIBAGO", "SAN NICOLAS", "SANTA TERESITA", "CLARO M. RECTO", "CUAYAN"])
    YEAR_OPTIONS = ["ALL"] + sorted(["2017","2018","2019","2020","2021","2022","2023", "2024"], reverse=True)
    MONTH_OPTIONS_VIZ = ["ALL", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    WEEKDAY_OPTIONS_VIZ = ["ALL", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    TIME_OF_DAY_OPTIONS_VIZ = ["ALL", "Morning", "Afternoon", "Evening", "Midnight"]


def filter_dataframe(crime_list, year_list, month_list=None, weekday_list=None,time_of_day_list=None, start_date=None, end_date=None):
    """Filters the main DataFrame based on user selections with multiple filters."""
    # Use retrained data if available, otherwise use original df
    global model_source_df
    source_df = model_source_df if not model_source_df.empty else df

    if source_df.empty:
        return pd.DataFrame()

    filtered_df = source_df.copy()

    # Filter by Crime Type (multi-select)
    if crime_list and "ALL" not in crime_list:
        filtered_df = filtered_df[filtered_df[COL_CRIME_DESC].isin(crime_list)]

    # Filter by Year (multi-select)
    if year_list and "ALL" not in year_list:
        # Convert year_list to match the dataframe's Year column type
        # Handle both string and int types in the dataframe
        year_values = []
        for y in year_list:
            if y != "ALL":
                try:
                    # Try to convert to int first
                    year_values.append(int(y))
                    # Also add string version in case df has string years
                    year_values.append(str(y))
                except (ValueError, TypeError):
                    pass
        if year_values:
            filtered_df = filtered_df[filtered_df[COL_YEAR].isin(year_values)]

    # Filter by Month (multi-select)
    if month_list and "ALL" not in month_list:
        # Convert month names to numbers
        month_numbers = [MONTH_NAME_TO_NUMBER[m] for m in month_list if m in MONTH_NAME_TO_NUMBER]
        filtered_df = filtered_df[filtered_df['Month'].isin(month_numbers)]

    # Filter by Weekday (multi-select)
    if weekday_list and "ALL" not in weekday_list:
        # Map weekday names to numbers (assuming Day_of_Week column has format like "Monday")
        filtered_df = filtered_df[filtered_df['Day_of_Week'].isin(weekday_list)]

    # Filter by Time of Day (multi-select)
    if time_of_day_list and "ALL" not in time_of_day_list:
        filtered_df = filtered_df[filtered_df['Time_of_Day'].isin(time_of_day_list)]

    # Filter by Date Range
    if start_date:
        filtered_df = filtered_df[filtered_df[COL_DATE] >= start_date]
    if end_date:
        filtered_df = filtered_df[filtered_df[COL_DATE] <= end_date]

    return filtered_df

def create_map(filtered_df):
    """Generates the Plotly Scatter Map figure (using MapLibre)."""
    if filtered_df.empty or filtered_df[[COL_LAT, COL_LON]].isnull().all().all():
        fig = go.Figure(go.Scattermap(lat=[15.1667], lon=[120.5833], mode='markers'))
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lon=120.5833, lat=15.1667),
                zoom=11
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            title="No data to display"
        )
        return fig

    fig = px.scatter_map(
        filtered_df,
        lat=COL_LAT,
        lon=COL_LON,
        color=COL_CRIME_DESC,
        hover_name=COL_CRIME_DESC,
        hover_data={
            COL_DATE: True,
            COL_AREA: True,
            COL_STATUS: True,
            COL_LAT: True,
            COL_LON: True,
            COL_CRIME_DESC: True,
            COL_AGE: True,
            COL_MALE_VICTIMS: True,
            COL_FEMALE_VICTIMS: True
        },
        zoom=12,
        title="Crime Hotspot Map",
        map_style="open-street-map"
    )

    fig.update_traces(marker=dict(size=10))

    fig.update_layout(
        margin={"r":5,"t":35,"l":0,"b":0},
        legend=dict(
            orientation="v",  
            yanchor="middle",
            y=0.5,
            xanchor="right",  
            x=0.99, 
            bgcolor="rgba(255, 255, 255, 0.8)", 
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        height=575,
        title_x=0.5,
        title_font=dict(size=20)
    )
    return fig

def create_forecast_chart(df, selected_barangays=None):
    """
    Generates a forecast using XGBoost Regressor.
    Includes data labels and improved aesthetics.
    """
    # 1. Filter Data
    target_df = df.copy()
    if selected_barangays and "ALL" not in selected_barangays:
        target_df = target_df[target_df['Barangay'].isin(selected_barangays)]

    if target_df.empty:
        return go.Figure(layout=go.Layout(title="Not enough data for forecasting"))

    # 2. Aggregate by Month
    target_df['Date'] = pd.to_datetime(target_df['Date'], dayfirst=True, errors='coerce')
    target_df = target_df.dropna(subset=['Date'])

    monthly_crimes = target_df.set_index('Date').resample('ME').size().reset_index(name='Count')

    if len(monthly_crimes) < 12:
         return go.Figure(layout=go.Layout(title="Insufficient data (Need > 12 months)"))

    # 3. Feature Engineering
    df_train = monthly_crimes.copy()
    df_train['Month'] = df_train['Date'].dt.month
    df_train['Year'] = df_train['Date'].dt.year
    df_train['Time_Index'] = np.arange(len(df_train))

    # Historical Average & Lag
    monthly_averages = df_train.groupby('Month')['Count'].mean()
    df_train['Historical_Month_Avg'] = df_train['Month'].map(monthly_averages)
    df_train['Lag_12'] = df_train['Count'].shift(12)

    df_train_clean = df_train.dropna(subset=['Lag_12'])

    # 4. Train XGBoost
    xgb_forecast = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror',
        random_state=42
    )

    features = ['Month', 'Year', 'Time_Index', 'Historical_Month_Avg', 'Lag_12']
    X = df_train_clean[features]
    y = df_train_clean['Count']

    xgb_forecast.fit(X, y)

    # 5. Generate Future Data
    last_date = monthly_crimes['Date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]

    future_data = pd.DataFrame({
        'Date': future_dates,
        'Month': [d.month for d in future_dates],
        'Year': [d.year for d in future_dates],
        'Time_Index': np.arange(len(df_train), len(df_train) + 12)
    })

    future_data['Historical_Month_Avg'] = future_data['Month'].map(monthly_averages)
    last_12_months_counts = monthly_crimes.iloc[-12:]['Count'].values
    future_data['Lag_12'] = last_12_months_counts

    # Predict
    future_counts = xgb_forecast.predict(future_data[features])
    future_counts = [max(0, int(x)) for x in future_counts]

    # 6. Create Plotly Figure
    fig = go.Figure()

    # A. Historical Line
    display_history = monthly_crimes.tail(24)

    fig.add_trace(go.Scatter(
        x=display_history['Date'],
        y=display_history['Count'],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#1a73e8', width=2.5),
        marker=dict(size=7, symbol='circle', line=dict(width=1, color='white')),
        hovertemplate='<b>Date:</b> %{x|%b %Y}<br><b>Crimes:</b> %{y}<extra></extra>'
    ))

    # B. Forecast Line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_counts,
        mode='lines+markers+text',
        name='Forecast (Next 12 Months)',
        line=dict(color='#d32f2f', width=2.5),
        marker=dict(size=8, symbol='circle', color='#d32f2f', line=dict(width=1, color='white')),
        text=future_counts,
        textposition='top center',
        textfont=dict(family="Arial", size=11, color="#d32f2f"),
        hovertemplate='<b>Date:</b> %{x|%b %Y}<br><b>Predicted:</b> %{y}<extra></extra>'
    ))

    # C. Title Content (Just the text, no tags)
    title_text = "Next 12 Months Projection"

    if selected_barangays and "ALL" not in selected_barangays:
        if len(selected_barangays) <= 2:
             title_text += f" ({', '.join(selected_barangays)})"

    # D. Layout styling
    fig.update_layout(
        title=dict(
            text=title_text,
            y=0.95,        
            x=0.5,           
            xanchor='center',
            yanchor='top',
            font=dict(  
                family="Arial, sans-serif",
                size=24,    
                color="black"
            )
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False
        ),
        yaxis=dict(
            title="Crime Count",
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, b=40, l=60, r=40), # Increased top margin for larger title
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig

def create_table(filtered_df):
    """Generates the Plotly Table for crime details."""
    cols_to_show = [
        COL_DATE,
        COL_AREA,
        COL_OFFENSE,
        COL_STATUS,
        COL_AGE,
        COL_MALE_VICTIMS,
        COL_FEMALE_VICTIMS
    ]
    display_names = {
        COL_DATE: "Date",
        COL_AREA: "Barangay",
        COL_OFFENSE: "Offense Committed",
        COL_STATUS: "Case Status",
        COL_AGE: "Victim's Age",
        COL_MALE_VICTIMS: "Male Victims",
        COL_FEMALE_VICTIMS: "Female Victims"
    }

    cols_exist = [col for col in cols_to_show if col in filtered_df.columns]

    if not cols_exist or filtered_df.empty:
        return go.Figure(go.Table(header=dict(values=['No Data']), cells=dict(values=[[]])))

    table_data = filtered_df[cols_exist].head(50)
    # Get the display-friendly header names and wrap them in <b> tags for bolding
    header_names = [f"<b>{display_names.get(col, col)}</b>" for col in cols_exist]

    # Create the Plotly Table
    fig = go.Figure(data=[go.Table(
        header=dict(values=header_names,
                    fill_color='paleturquoise',
                    align='center',
                    font=dict(size=14)),
        cells=dict(values=[table_data[col] for col in cols_exist],
                   fill_color='lavender',
                   align='center'))
    ])

    fig.update_layout(
        title="Crime Details (First 50 Entries)",
        margin={"r":5,"t":30,"l":5,"b":5},
        title_x=0.5,
        title_font=dict(size=20)
    )
    return fig

def create_pie(filtered_df):
    """Generates the Plotly Pie Chart for victim age groups."""
    if filtered_df.empty:
        return go.Figure(data=[go.Pie(labels=[], values=[])], layout=go.Layout(title="Victim Age Groups (No Data)"))
    age_cols_exist = [col for col in VICTIM_AGE_COLS if col in filtered_df.columns]
    if not age_cols_exist:
        return go.Figure(data=[go.Pie(labels=[], values=[])], layout=go.Layout(title="Victim Age Groups (No Data)"))

    age_sums = filtered_df[age_cols_exist].sum()
    age_sums = age_sums[age_sums > 0]

    if age_sums.empty:
         return go.Figure(data=[go.Pie(labels=[], values=[])], layout=go.Layout(title="Victim Age Groups (No Data)"))

    age_labels = [col.replace('Victims_', '').replace('_', ' ') for col in age_sums.index]

    fig = go.Figure(data=[go.Pie(
        labels=age_labels,
        values=age_sums.values,
        hole=.3,
        textinfo='percent+label'
    )])
    fig.update_layout(
        title_text='Victim Age Groups',
        margin=dict(t=50, b=20, l=20, r=20),
        height=375,
        title_x=0.5,
        title_font=dict(size=25)
    )
    return fig

def create_time_of_day_bar(filtered_df):
    """Generates the Plotly Bar Chart for time of day."""
    if filtered_df.empty or COL_TIME_OF_DAY not in filtered_df.columns:
        return go.Figure(layout=go.Layout(title="Crimes by Time of Day (No Data)"))

    time_of_day_order = ['Morning', 'Afternoon', 'Evening', 'Midnight']
    time_counts = filtered_df[COL_TIME_OF_DAY].value_counts().reindex(time_of_day_order).fillna(0)

    fig = px.bar(
        time_counts,
        x=time_counts.index,
        y=time_counts.values,
        labels={'y': 'Number of Crimes', 'x': 'Time of Day'},
        title='Crime Counts by Time of Day'
    )
    fig.update_layout(
        margin=dict(t=50, b=40, l=40, r=20),
        height=375,
        title_x=0.5,
        title_font=dict(size=25)
    )
    return fig

def create_day_of_week_bar(filtered_df):
    """Generates the Plotly Bar Chart for day of the week."""
    if filtered_df.empty or COL_DAY_OF_WEEK not in filtered_df.columns:
        return go.Figure(layout=go.Layout(title="Crimes by Day of Week (No Data)"))

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = filtered_df[COL_DAY_OF_WEEK].value_counts().reindex(day_order).fillna(0)

    fig = px.bar(
        day_counts,
        x=day_counts.index,
        y=day_counts.values,
        labels={'y': 'Number of Crimes', 'x': 'Day of Week'},
        title='Crimes by Day of Week'
    )
    fig.update_layout(
        margin=dict(t=50, b=40, l=40, r=20),
        height=375,
        title_x=0.5,
        title_font=dict(size=25)
    )
    return fig

def create_line_plot(base_df, selected_crime):
    """Generates the Plotly Line Chart for crime trends over years."""
    if base_df.empty or COL_YEAR not in base_df.columns:
        return go.Figure(layout=go.Layout(title="Crime Trend Over Years (No Data)"))

    if selected_crime == "ALL":
        trend_df = base_df.groupby(COL_YEAR).size().reset_index(name='Count')
        title = "Total Crime Trend Over Years"
    else:
        trend_df = base_df[base_df[COL_CRIME_DESC] == selected_crime].groupby(COL_YEAR).size().reset_index(name='Count')
        title = f'"{selected_crime}" Trend Over Years'

    if trend_df.empty:
        return go.Figure(layout=go.Layout(title=f"Crime Trend Over Years (No Data for {selected_crime})"))

    trend_df = trend_df.sort_values(by=COL_YEAR)

    fig = px.line(
        trend_df,
        x=COL_YEAR,
        y='Count',
        title=title,
        markers=True
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        margin=dict(t=50, b=40, l=40, r=20),
        height=375,
        title_x=0.5,
        title_font=dict(size=25)
    )
    return fig

def aggregate(df):
    # Ensure all required columns exist
    required_cols = [
        'Barangay', 'Month', 'Weekday', 'Time_of_Day', 'Offense ID', 'Hour',
        'Is_Weekend', 'Population_2024', 'Pop_Density_2024', 'Area_sqkm',
        'Distance_from_Police', 'Num_Police_Stations_1km', 'Num_Victims',
        'Num_Suspects', 'Focus_Crime'
    ]

   # Check for columns that might be missing
    df_columns = set(df.columns)
    missing_cols = [col for col in required_cols if col not in df_columns]

    if missing_cols:
        print(f"Error: Missing required columns for aggregation: {missing_cols}")
        # Create dummy columns with 0 or np.nan to avoid crashing
        for col in missing_cols:
            df[col] = 0
        print("Warning: Missing columns filled with 0.")

    # DEBUG: Check data quality before aggregation
    print(f"\nData Quality Check BEFORE aggregation:")
    print(f"   Total rows: {len(df)}")
    print(f"   Rows with missing Weekday: {df['Weekday'].isna().sum()}")
    print(f"   Rows with missing Time_of_Day: {df['Time_of_Day'].isna().sum()}")
    print(f"   Rows with missing Month: {df['Month'].isna().sum()}")

    # Check BALIBAGO specifically
    balibago_before = df[(df['Barangay'] == 'BALIBAGO') & (df['Month'] == 1) & (df['Day_of_Week'] == 'Monday')]
    print(f"   BALIBAGO, January, Monday rows BEFORE aggregation: {len(balibago_before)}")
    if len(balibago_before) > 0:
        print(f"   Time_of_Day distribution: {balibago_before['Time_of_Day'].value_counts().to_dict()}")
        print(f"   Weekday values: {balibago_before['Weekday'].unique()}")

    # Clean data: Remove rows with missing critical grouping columns
    df_clean = df.dropna(subset=['Barangay', 'Month', 'Weekday', 'Time_of_Day'])
    rows_dropped = len(df) - len(df_clean)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with missing grouping columns")

    # Group on 'Weekday' (numeric 1-7), not 'Day_of_Week' (text), to match xgboost_model.py
    agg_df = df_clean.groupby(['Barangay', 'Month', 'Weekday', 'Time_of_Day']).agg(
        Crime_Count=(COL_OFFENSE_ID, 'count'),
        Avg_Hour = (COL_HOUR, 'mean'),
        Mode_Hour=(COL_HOUR, lambda x: x.mode()[0] if not x.mode().empty else 0),
        Mode_Focus_Crime=(COL_FOCUS_CRIME, lambda x: x.mode()[0] if not x.mode().empty else 'N/A'),
        Weekend_Crimes=(COL_IS_WEEKEND, 'sum'),
        Weekday_Crimes=(COL_IS_WEEKEND, lambda x: (~x.astype(bool)).sum()),
        Population=(COL_POPULATION, 'first'),
        Pop_Density=(COL_POP_DENSITY, 'first'),
        Area_sqkm=(COL_AREA_SQKM, 'first'),
        Avg_Distance_Police=(COL_DISTANCE_POLICE, 'mean'),
        Avg_Num_Stations_1km=(COL_NUM_STATIONS_1KM, 'mean'),
        Avg_Victims=(COL_NUM_VICTIMS, 'median'),
        Avg_Suspects=(COL_NUM_SUSPECTS, 'median'),
        Murder_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Murder').sum()),
        Theft_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Theft').sum()),
        Robbery_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Robbery').sum()),
        Physical_Injuries_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Physical Injuries').sum()),
        Rape_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Rape').sum()),
        Homicide_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Homicide').sum()),
        Carnapping_MC_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Carnapping MC').sum()),
        Carnapping_MV_Count=(COL_FOCUS_CRIME, lambda x: (x == 'Carnapping MV').sum())
    ).reset_index()

    agg_df['Crime_Rate_per_1000'] = (agg_df['Crime_Count'] / agg_df['Population']) * 1000
    agg_df['Crime_Density_sqkm'] = agg_df['Crime_Count'] / agg_df['Area_sqkm']
    agg_df['Weekend_Ratio'] = agg_df['Weekend_Crimes'] / (agg_df['Crime_Count'] + 1e-6)

    # Sort by the same columns used in groupby
    agg_df = agg_df.sort_values(['Barangay', 'Month', 'Weekday', 'Time_of_Day'])
    agg_df['Crime_Count_Lag1'] = agg_df.groupby('Barangay')['Crime_Count'].shift(1)
    agg_df['Crime_Count_Lag2'] = agg_df.groupby('Barangay')['Crime_Count'].shift(2)
    agg_df['Crime_Count_Rolling_3m'] = agg_df.groupby('Barangay')['Crime_Count'].rolling(3).mean().reset_index(drop=True)

    # DEBUG: Check BALIBAGO after aggregation but before dropna
    balibago_after = agg_df[(agg_df['Barangay'] == 'BALIBAGO') & (agg_df['Month'] == 1) & (agg_df['Weekday'] == 1)]
    print(f"\nAfter aggregation (before dropna):")
    print(f"   BALIBAGO, January, Weekday=1 rows: {len(balibago_after)}")
    if len(balibago_after) > 0:
        print(f"   Time_of_Day values: {balibago_after['Time_of_Day'].tolist()}")
        print(f"   Crime counts: {balibago_after['Crime_Count'].tolist()}")
        # Check for NaN values
        print(f"   Rows with ANY NaN: {balibago_after.isna().any(axis=1).sum()}")
        if balibago_after.isna().any(axis=1).sum() > 0:
            print(f"   Columns with NaN in these rows: {balibago_after.columns[balibago_after.isna().any()].tolist()}")

    # Only drop rows missing the 7 features the model actually uses
    prediction_features = [
        'Population', 'Area_sqkm', 'Avg_Num_Stations_1km',
        'Weekend_Ratio', 'Avg_Hour', 'Avg_Victims', 'Avg_Suspects'
    ]

    rows_before_dropna = len(agg_df)
    agg_df = agg_df.dropna(subset=prediction_features)
    rows_after_dropna = len(agg_df)
    rows_dropped_by_dropna = rows_before_dropna - rows_after_dropna

    if rows_dropped_by_dropna > 0:
        print(f"Dropped {rows_dropped_by_dropna} rows missing prediction features")

    # DEBUG: Check BALIBAGO after dropna
    balibago_final = agg_df[(agg_df['Barangay'] == 'BALIBAGO') & (agg_df['Month'] == 1) & (agg_df['Weekday'] == 1)]
    print(f"\nAfter dropna (FINAL):")
    print(f"   BALIBAGO, January, Weekday=1 rows: {len(balibago_final)}")
    if len(balibago_final) > 0:
        print(f"   Time_of_Day values: {balibago_final['Time_of_Day'].tolist()}")
        print(f"   Crime counts: {balibago_final['Crime_Count'].tolist()}")

    print(f"\nAggregated dataset shape: {agg_df.shape}")
    print(f"   Original training shape was (2505, 30)")
    if len(agg_df) > 2505:
        extra_rows = len(agg_df) - 2505
        print(f"   {extra_rows} additional rows preserved (had NaN in unused lag features)")
        print(f"   This is GOOD - we now correctly show historical crime data that was mistakenly excluded!")
    return agg_df


try:
    model_data_path = 'data/focus_df.csv'
    model_source_df = pd.read_csv(model_data_path, parse_dates=['Date'])
    print("Loading and aggregating data for prediction model (matching xgboost_model.py)...")

    # Aggregate the entire dataset, unfiltered by date, to match xgboost_model.py
    # This matches the training approach: train_df = focus_df
    print("Aggregating entire dataset...")
    AGGREGATED_DF = aggregate(model_source_df)

    print(f"Aggregation complete. Final lookup table shape: {AGGREGATED_DF.shape}")
    print(f"   Expected shape: (2505, 30) - matches xgboost_model.py training data")

    # Print sample of aggregated data for debugging
    print("\nSample of AGGREGATED_DF (first 3 rows):")
    print(AGGREGATED_DF[['Barangay', 'Month', 'Weekday', 'Time_of_Day', 'Crime_Count']].head(3))

    # Check for BALIBAGO specifically
    balibago_data = AGGREGATED_DF[(AGGREGATED_DF['Barangay'] == 'BALIBAGO') &
                                   (AGGREGATED_DF['Month'] == 1) &
                                   (AGGREGATED_DF['Weekday'] == 1)]
    print(f"\nBALIBAGO data for January, Monday (Weekday=1): {len(balibago_data)} rows found")
    if not balibago_data.empty:
        print(balibago_data[['Barangay', 'Month', 'Weekday', 'Time_of_Day', 'Crime_Count']])

    # Create a barangay demographics lookup (Population and Area for each barangay)
    BARANGAY_DEMOGRAPHICS = {}
    if not AGGREGATED_DF.empty:
        for barangay in AGGREGATED_DF['Barangay'].unique():
            barangay_data = AGGREGATED_DF[AGGREGATED_DF['Barangay'] == barangay].iloc[0]
            BARANGAY_DEMOGRAPHICS[barangay] = {
                'population': int(barangay_data['Population']),
                'area_sqkm': float(barangay_data['Area_sqkm'])
            }
    print(f"Created demographics lookup for {len(BARANGAY_DEMOGRAPHICS)} barangays")

except FileNotFoundError:
    print(f"Error: '{model_data_path}' not found. Model predictions will be disabled.")
    model_source_df = pd.DataFrame()
    AGGREGATED_DF = pd.DataFrame()
    BARANGAY_DEMOGRAPHICS = {}
except Exception as e:
    print(f"Error loading or aggregating model data: {e}")
    model_source_df = pd.DataFrame()
    AGGREGATED_DF = pd.DataFrame()
    BARANGAY_DEMOGRAPHICS = {}

try:
    MODEL = joblib.load('pulisai_xgb_model.joblib')
    print("XGBoost model loaded successfully.")
except FileNotFoundError:
    print("Error: 'pulisai_xgb_model.joblib' not found.")
    MODEL = None

# Initialize alarm thresholds (will be updated during retraining)
ALARM_THRESHOLDS = {'q25': None, 'q75': None}

# Features used by the model
SELECTED_FEATURES = [
    'Population',
    'Area_sqkm',
    'Avg_Num_Stations_1km',
    'Weekend_Ratio',
    'Avg_Hour',
    'Avg_Victims',
    'Avg_Suspects'
]

# Map for converting model output (0, 1, 2) back to text
ALARM_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}
# For cases with no historical data, barangays are classified as 'Low' (green color)

# Create options for prediction dropdowns
if not model_source_df.empty:
    # Convert month numbers to names
    month_numbers = sorted(model_source_df['Month'].unique())
    MONTH_OPTIONS = [MONTH_NAMES[m] for m in month_numbers if m in MONTH_NAMES]
    WEEKDAY_OPTIONS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    TIME_OF_DAY_OPTIONS = ['Morning', 'Afternoon', 'Evening', 'Midnight']
else:
    MONTH_OPTIONS = [MONTH_NAMES[m] for m in range(1, 13)]
    WEEKDAY_OPTIONS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    TIME_OF_DAY_OPTIONS = ['Morning', 'Afternoon', 'Evening', 'Midnight']


#Main Flask Routes

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('crimehotspot'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('crimehotspot'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate credentials
        if username in VALID_USERS and VALID_USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('crimehotspot'))
        else:
            error = 'Invalid credentials. Please check your username and password and try again.'

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/graph')
@login_required
def graph():
    """
    This route now generates all the charts and passes them to the template.
    """
    # Get filter data from URL - multi-select parameters
    crime_list = request.args.getlist('crime')
    year_list = request.args.getlist('year')
    month_list = request.args.getlist('month')
    weekday_list = request.args.getlist('weekday')
    time_of_day_list = request.args.getlist('time_of_day')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    # Handle empty or missing selections (default to ALL)
    if not crime_list or crime_list == ['']:
        crime_list = ['ALL']
    if not year_list or year_list == ['']:
        year_list = ['ALL']
    if not month_list or month_list == ['']:
        month_list = ['ALL']
    if not weekday_list or weekday_list == ['']:
        weekday_list = ['ALL']
    if not time_of_day_list or time_of_day_list == ['']:
        time_of_day_list = ['ALL']

    # Convert empty date strings to None
    start_date = start_date if start_date else None
    end_date = end_date if end_date else None

    # Apply filters
    filtered_df = filter_dataframe(
        crime_list, year_list, month_list, weekday_list,
        time_of_day_list, start_date, end_date
    )

    # Get min/max dates from the current dataset (use retrained data if available)
    global model_source_df
    current_df = model_source_df if not model_source_df.empty else df

    if not current_df.empty and 'Date' in current_df.columns:
        try:
            # Convert to datetime if not already
            date_col = pd.to_datetime(current_df['Date'])
            min_date_limit = date_col.min().strftime('%Y-%m-%d')
            max_date_limit = date_col.max().strftime('%Y-%m-%d')
        except:
            min_date_limit = '2017-01-01'
            max_date_limit = '2025-12-31'
    else:
        min_date_limit = '2017-01-01'
        max_date_limit = '2025-12-31'

    # Calculate Totals
    total_crimes = len(filtered_df)
    total_solved = 0
    total_suspects = 0
    total_victims = 0
    top_barangay = "N/A"
    top_crime = "N/A"
    total_underinvestigation = 0
    total_cleared = 0

    if not filtered_df.empty:
        if COL_STATUS in filtered_df.columns:
            total_solved = len(filtered_df[filtered_df[COL_STATUS] == 'Solved'])
            total_underinvestigation = len(filtered_df[filtered_df[COL_STATUS] == 'Under Investigation'])
            total_cleared = len(filtered_df[filtered_df[COL_STATUS] == 'Cleared'])
        # Calculate total suspects and victims
        if 'Num_Suspects' in filtered_df.columns:
            total_suspects = int(filtered_df['Num_Suspects'].sum())
        if 'Num_Victims' in filtered_df.columns:
            total_victims = int(filtered_df['Num_Victims'].sum())

        # Calculate Top Barangay (most crimes) - show name only in proper case
        if COL_AREA in filtered_df.columns:
            barangay_counts = filtered_df[COL_AREA].value_counts()
            if not barangay_counts.empty:
                top_barangay = barangay_counts.index[0].title()

        # Calculate Top Crime (most frequent) - show name only
        if COL_CRIME_DESC in filtered_df.columns:
            crime_counts = filtered_df[COL_CRIME_DESC].value_counts()
            if not crime_counts.empty:
                top_crime = crime_counts.index[0]

    # Generate all chart figures
    map_fig = create_map(filtered_df)
    table_fig = create_table(filtered_df)
    pie_fig = create_pie(filtered_df)
    bar_fig = create_time_of_day_bar(filtered_df)
    day_bar_fig = create_day_of_week_bar(filtered_df)
    # For line plot, pass first crime if available, otherwise 'ALL'
    line_crime = crime_list[0] if crime_list and crime_list[0] != 'ALL' else 'ALL'
    line_fig = create_line_plot(df, line_crime)

    responsive_config = {'responsive': True}

    # Convert Plotly figures to HTML
    map_html = map_fig.to_html(full_html=False, include_plotlyjs='cdn', config=responsive_config)
    table_html = table_fig.to_html(full_html=False, include_plotlyjs=False, default_height='100%', config=responsive_config)
    pie_html = pie_fig.to_html(full_html=False, include_plotlyjs=False, config=responsive_config)
    bar_html = bar_fig.to_html(full_html=False, include_plotlyjs=False,config=responsive_config)
    day_bar_html = day_bar_fig.to_html(full_html=False, include_plotlyjs=False,config=responsive_config)
    line_html = line_fig.to_html(full_html=False, include_plotlyjs=False,config=responsive_config)

    return render_template(
        'graph.html',
        active='graph',
        CRIME_OPTIONS=CRIME_OPTIONS,
        YEAR_OPTIONS=YEAR_OPTIONS,
        MONTH_OPTIONS_VIZ=MONTH_OPTIONS_VIZ,
        WEEKDAY_OPTIONS_VIZ=WEEKDAY_OPTIONS_VIZ,
        TIME_OF_DAY_OPTIONS_VIZ=TIME_OF_DAY_OPTIONS_VIZ,
        crime_selected=crime_list,
        year_selected=year_list,
        month_selected=month_list,
        weekday_selected=weekday_list,
        time_of_day_selected=time_of_day_list,
        start_date=start_date or '',
        end_date=end_date or '',
        min_date_limit=min_date_limit,
        max_date_limit=max_date_limit,

        total_crimes=total_crimes,
        total_solved=total_solved,
        total_cleared=total_cleared,
        total_suspects=total_suspects,
        total_victims=total_victims,
        total_underinvestigation=total_underinvestigation,
        top_barangay=top_barangay,
        top_crime=top_crime,
        map_html=map_html,
        table_html=table_html,
        pie_html=pie_html,
        bar_html=bar_html,
        day_bar_html=day_bar_html,
        line_html=line_html
    )

@app.route('/crimehotspot', methods=['GET', 'POST'])
@login_required
def crimehotspot():

    predictions_list = []
    barangay_predictions = {}  # Dictionary to store all barangay predictions

    # Set default selections
    barangay_selected = []
    month_selected = []
    month_selected_names = []  # For display
    weekday_selected = None
    time_of_day_selected = None

    has_prediction = False  # Track if we've made a prediction
    forecast_html = None    # Hidden until a prediction is made

    if request.method == 'POST':
        # Get selections from form
        barangay_selected = request.form.getlist('barangay')
        month_selected_names = request.form.getlist('month')  # These are now month names
        weekday_selected = request.form.get('weekday')
        time_of_day_selected = request.form.get('time_of_day')

        # Convert month names to numbers for processing
        month_selected = [MONTH_NAME_TO_NUMBER[m] for m in month_selected_names if m in MONTH_NAME_TO_NUMBER]

        # If nothing selected, set defaults
        if not barangay_selected:
            barangay_selected = [BARANGAY_OPTIONS[1]] if len(BARANGAY_OPTIONS) > 1 else []
        if not month_selected:
            month_selected = [MONTH_NAME_TO_NUMBER[MONTH_OPTIONS[0]]]
            month_selected_names = [MONTH_OPTIONS[0]]
        if not weekday_selected:
            weekday_selected = WEEKDAY_OPTIONS[0]
        if not time_of_day_selected:
            time_of_day_selected = TIME_OF_DAY_OPTIONS[0]

        has_prediction = True

    # Generate predictions for selected combinations (MAP LOGIC)
    if request.method == 'POST' and MODEL is not None and not AGGREGATED_DF.empty:
        try:
            # Map weekday names to the numeric values used in the lookup table (1=Monday, 7=Sunday)
            weekday_map = {
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }

            # Iterate through all combinations
            for barangay in barangay_selected:
                for month in month_selected:
                    try:
                        time_of_day = time_of_day_selected
                        month_int = int(month)
                        weekday_int = weekday_map.get(weekday_selected, 1)  # Convert to numeric

                        # Find matching row in aggregated data (using 'Weekday' numeric column)
                        row = AGGREGATED_DF.loc[
                            (AGGREGATED_DF['Barangay'] == barangay) &
                            (AGGREGATED_DF['Month'] == month_int) &
                            (AGGREGATED_DF['Weekday'] == weekday_int) &
                            (AGGREGATED_DF['Time_of_Day'] == time_of_day)
                        ]

                        if not row.empty:
                            features_df = row.iloc[[0]][SELECTED_FEATURES]
                            prediction_raw = MODEL.predict(features_df)
                            alarm_level = ALARM_MAP[prediction_raw[0]]

                            # Get all the data from the aggregated row
                            mode_hour = row.iloc[0]['Mode_Hour']
                            mode_crime = row.iloc[0]['Mode_Focus_Crime']
                            crime_density = row.iloc[0]['Crime_Density_sqkm']
                            population = row.iloc[0]['Population']
                            area_sqkm = row.iloc[0]['Area_sqkm']

                            # Store an object with all data points
                            barangay_predictions[barangay] = {
                                "level": alarm_level,
                                "mode_hour": int(mode_hour),
                                "mode_crime": str(mode_crime),
                                "crime_density": float(crime_density),
                                "population": int(population),
                                "area_sqkm": float(area_sqkm)
                            }

                        else:
                            # No data for this combination - treat as Low risk
                            if barangay not in barangay_predictions:
                                # Get demographics from lookup if available
                                demographics = BARANGAY_DEMOGRAPHICS.get(barangay, {})
                                population = demographics.get('population', 'N/A')
                                area_sqkm = demographics.get('area_sqkm', 'N/A')

                                # Calculate crime density as 0 since there's no historical crime for this specific combination
                                crime_density = 0.0 if area_sqkm != 'N/A' else 'N/A'

                                barangay_predictions[barangay] = {
                                    "level": "Low",
                                    "mode_hour": "N/A",
                                    "mode_crime": "N/A",
                                    "crime_density": crime_density,
                                    "population": population,
                                    "area_sqkm": area_sqkm
                                }
                    except Exception as e:
                        print(f"Error predicting for {barangay}: {e}")

        except Exception as e:
            print(f"Error during prediction: {e}")
    elif request.method == 'POST':
        predictions_list = [{'barangay': 'Error', 'alarm': 'Error', 'details': 'Model is not loaded'}]


    #FORECAST LOGIC (Only runs if Predict was clicked)
    if has_prediction:
        global model_source_df
        # Use selected barangays
        forecast_barangays = barangay_selected if barangay_selected else ["ALL"]

        forecast_fig = create_forecast_chart(model_source_df, forecast_barangays)

        # Convert to HTML
        responsive_config = {'responsive': True}
        forecast_html = forecast_fig.to_html(full_html=False, include_plotlyjs='cdn', config=responsive_config)


    # Load GeoJSON data
    geojson_data = None
    try:
        with open('angeles_city_barangays.geojson', 'r') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        print("GeoJSON file not found")


    return render_template(
        'crimehotspot.html',
        active='crimehotspot',
        # Options for plotting dropdowns (from base.html)
        CRIME_OPTIONS=CRIME_OPTIONS,
        YEAR_OPTIONS=YEAR_OPTIONS,
        forecast_html=forecast_html, # Will be None on initial load

        # Options for prediction dropdowns
        BARANGAY_OPTIONS_PREDICT=BARANGAY_OPTIONS, # Use the same list
        MONTH_OPTIONS_PREDICT=MONTH_OPTIONS,
        WEEKDAY_OPTIONS_PREDICT=WEEKDAY_OPTIONS,
        TIME_OF_DAY_OPTIONS_PREDICT=TIME_OF_DAY_OPTIONS,

        # Pass selected values back to the form
        barangay_selected=barangay_selected if has_prediction else [],
        month_selected=month_selected_names if has_prediction else [],
        weekday_selected=weekday_selected if has_prediction else None,
        time_of_day_selected=time_of_day_selected if has_prediction else None,

        # Map data - ALWAYS pass geojson_data so map shows on initial load
        geojson_data=json.dumps(geojson_data) if geojson_data else None,
        barangay_predictions=json.dumps(barangay_predictions) if has_prediction else json.dumps({}),
        barangay_name_mapping=json.dumps(BARANGAY_NAME_MAPPING),
        selected_barangays=json.dumps(barangay_selected) if has_prediction else json.dumps([])
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Handle file upload and model retraining."""
    global MODEL, AGGREGATED_DF, model_source_df, ALARM_THRESHOLDS

    # Get current focus_df for display
    current_focus_df = model_source_df if model_source_df is not None else df

    # Select columns for display
    display_columns = ['Offense ID', 'Barangay', 'Date', 'Time Committed', 'Offense Committed',
                      'Focus_Crime', 'Crime Type', 'Case Status', 'Year', 'Month', 'Weekday', 'Time_of_Day']
    focus_data = current_focus_df[display_columns].to_dict('records')

    if request.method == 'POST':
        try:
            # Check if files are present
            if 'crime_file' not in request.files or 'suspect_file' not in request.files or 'victim_file' not in request.files:
                return render_template(
                    'upload.html',
                    active='upload',
                    focus_data=focus_data,
                    error_message="Please upload all three required files (crime, suspect, victim)."
                )

            crime_file = request.files['crime_file']
            suspect_file = request.files['suspect_file']
            victim_file = request.files['victim_file']

            # Check if files are selected
            if crime_file.filename == '' or suspect_file.filename == '' or victim_file.filename == '':
                return render_template(
                    'upload.html',
                    active='upload',
                    focus_data=focus_data,
                    error_message="Please select all three files before uploading."
                )

            # Read uploaded files into DataFrames
            print("\n" + "="*60)
            print("READING UPLOADED FILES")
            print("="*60)

            crime_df_new = pd.read_csv(crime_file)
            suspect_df_new = pd.read_csv(suspect_file)
            victim_df_new = pd.read_csv(victim_file)

            print(f"   Crime data: {crime_df_new.shape}")
            print(f"   Suspect data: {suspect_df_new.shape}")
            print(f"   Victim data: {victim_df_new.shape}")

            # Load barangay info from data folder
            barangay_info_path = os.path.join('data', 'angeles_city_other_info.csv')
            if not os.path.exists(barangay_info_path):
                return render_template(
                    'upload.html',
                    active='upload',
                    focus_data=focus_data,
                    error_message=f"Barangay info file not found at {barangay_info_path}. Please ensure angeles_city_other_info.csv exists in data folder."
                )

            barangay_info = pd.read_csv(barangay_info_path)
            print(f"   Barangay info loaded: {barangay_info.shape}")

            # Process uploaded data through the pipeline
            print("\nStarting data processing pipeline...")
            focus_df_new = data_processor.process_uploaded_data(
                crime_df_new, suspect_df_new, victim_df_new, barangay_info
            )

            # Load original 2017-2024 data
            print("\nLoading original 2017-2024 data...")
            focus_df_original = pd.read_csv('data/focus_df.csv', parse_dates=['Date'])
            print(f"   Original data loaded: {focus_df_original.shape}")

            # Combine original and new data
            print("\nCombining 2017-2024 and 2025 datasets...")
            focus_df_combined = pd.concat([focus_df_original, focus_df_new], ignore_index=True)
            print(f"   Combined dataset: {focus_df_combined.shape}")

            # Aggregate the combined data for training
            print("\nAggregating combined data for training...")
            train_df_old = data_processor.aggregate_for_training(focus_df_original.copy())
            train_df_new = data_processor.aggregate_for_training(focus_df_combined.copy())

            # Classify alarm levels for both datasets
            print("\nClassifying alarm levels (OLD - 2017-2024)...")
            train_df_old, old_q25, old_q75 = data_processor.classify_alarm_levels(train_df_old)
            old_alarm_dist = train_df_old['Alarm_Level'].value_counts().to_dict()

            print("\nClassifying alarm levels (NEW - 2017-2025)...")
            train_df_new, new_q25, new_q75 = data_processor.classify_alarm_levels(train_df_new)
            new_alarm_dist = train_df_new['Alarm_Level'].value_counts().to_dict()

            # Retrain the model
            print("\nRetraining XGBoost model...")
            success, model_path = retrain_model(train_df_new)

            if success:
                # Update global variables with new model and data
                MODEL = joblib.load(model_path)
                AGGREGATED_DF = train_df_new
                model_source_df = focus_df_combined
                ALARM_THRESHOLDS = {'q25': new_q25, 'q75': new_q75}

                # Update YEAR_OPTIONS, CRIME_OPTIONS, and BARANGAY_OPTIONS with new data
                global YEAR_OPTIONS, CRIME_OPTIONS, BARANGAY_OPTIONS
                if not focus_df_combined.empty:
                    if COL_YEAR in focus_df_combined.columns:
                        # Ensure Year column is string type before sorting
                        focus_df_combined[COL_YEAR] = focus_df_combined[COL_YEAR].astype(str)
                        sorted_years = sorted(focus_df_combined[COL_YEAR].unique(), reverse=True)
                        YEAR_OPTIONS = np.insert(sorted_years, 0, "ALL")

                    if COL_CRIME_DESC in focus_df_combined.columns:
                        sorted_crimes = sorted(focus_df_combined[COL_CRIME_DESC].unique())
                        CRIME_OPTIONS = np.insert(sorted_crimes, 0, "ALL")

                    if COL_AREA in focus_df_combined.columns:
                        sorted_barangays = sorted(focus_df_combined[COL_AREA].unique())
                        BARANGAY_OPTIONS = np.insert(sorted_barangays, 0, "ALL")

                print("   Global variables updated with retrained model")

                # Prepare statistics for display
                retrain_stats = {
                    'original_rows': len(focus_df_original),
                    'new_rows': len(focus_df_new),
                    'combined_rows': len(focus_df_combined),
                    'train_shape': train_df_new.shape,
                    'old_alarm': old_alarm_dist,
                    'new_alarm': new_alarm_dist,
                    'old_q25': round(old_q25, 2),
                    'old_q75': round(old_q75, 2),
                    'new_q25': round(new_q25, 2),
                    'new_q75': round(new_q75, 2)
                }

                print("\n" + "="*60)
                print("MODEL RETRAINING COMPLETED SUCCESSFULLY")
                print("="*60 + "\n")

                # Update focus_data with new combined data
                updated_focus_df = model_source_df if model_source_df is not None else df
                updated_focus_data = updated_focus_df[display_columns].to_dict('records')

                return render_template(
                    'upload.html',
                    active='upload',
                    focus_data=updated_focus_data,
                    success_message="Model retrained successfully with 2017-2025 data!",
                    retrain_stats=retrain_stats
                )
            else:
                return render_template(
                    'upload.html',
                    active='upload',
                    focus_data=focus_data,
                    error_message="Model retraining failed. Please check the logs."
                )

        except Exception as e:
            print(f"\nERROR during upload/retraining: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template(
                'upload.html',
                active='upload',
                focus_data=focus_data,
                error_message=f"An error occurred: {str(e)}"
            )

    return render_template(
        'upload.html',
        active='upload',
        focus_data=focus_data
    )


def retrain_model(train_df):
    """
    Retrain the XGBoost model with new data.

    Args:
        train_df: Aggregated training DataFrame with Alarm_Level

    Returns:
        success (bool), model_path (str)
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        print("   Preparing features and labels...")

        # Define features (same as original model)
        features = [
            'Population', 'Area_sqkm', 'Avg_Num_Stations_1km',
            'Weekend_Ratio', 'Avg_Hour', 'Avg_Victims', 'Avg_Suspects'
        ]

        X = train_df[features]
        y = train_df['Alarm_Level']

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        print(f"   Training data shape: X={X.shape}, y={y_encoded.shape}")
        print(f"   Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

        # Split data for evaluation
        print("\n   Splitting data for training and evaluation (75/25 split)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        print(f"      Training set: {X_train.shape[0]} samples")
        print(f"      Test set: {X_test.shape[0]} samples")

        # Train XGBoost model with SAME parameters as xgboost_model.py
        print("\n   Training XGBoost model with RandomizedSearchCV...")

        # Parameter grid matching xgboost_model.py exactly
        param_grid = {
            'max_depth': [5],
            'learning_rate': [0.05],
            'n_estimators': [400],
            'gamma': [0],
            'subsample': [0.9],
            'colsample_bytree': [0.7],
            'reg_alpha': [0.01],
            'reg_lambda': [1.5]
        }

        xgb_model = XGBClassifier(random_state=42)

        xgb_grid_search = RandomizedSearchCV(
            xgb_model, param_grid, cv=5, scoring="accuracy",
            n_jobs=-1, random_state=42, verbose=1
        )

        xgb_grid_search.fit(X_train, y_train)
        model = xgb_grid_search.best_estimator_

        # Get training and CV scores
        train_accuracy = model.score(X_train, y_train)
        cv_score = xgb_grid_search.best_score_

        # Evaluate model
        print("\n   Evaluating model performance...")
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        gap_score = train_accuracy - test_accuracy

        print(f"\n   Model Performance:")
        print(f"      Training Accuracy:      {train_accuracy:.4f}")
        print(f"      Cross-Validation Score: {cv_score:.4f}")
        print(f"      Test Accuracy (25%):    {test_accuracy:.4f}")
        print(f"      Gap Score:              {gap_score:.4f}", end="")

        if gap_score < 0.05:
            gap_label = 'Good Fit'
        elif gap_score >= 0.05 and gap_score < 0.15:
            gap_label = 'Overfit (Moderate)'
        else:
            gap_label = 'Underfit'

        print(f" ({gap_label})")

        # Classification Report
        print("\n   CLASSIFICATION REPORT (Including 2025 Data):")
        print("   " + "="*60)
        class_names = label_encoder.classes_  # ['High', 'Low', 'Medium']
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            zero_division=0
        )
        # Indent each line for better formatting
        for line in report.split('\n'):
            if line.strip():
                print(f"   {line}")
        print("   " + "="*60)

        # Confusion Matrix
        print("\n   Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"      Predicted →  {' '.join([f'{c:>8}' for c in class_names])}")
        for i, actual_class in enumerate(class_names):
            print(f"      {actual_class:>8}     {' '.join([f'{cm[i][j]:>8}' for j in range(len(class_names))])}")

        # Save model
        print("\n   Saving retrained model...")
        model_path = 'pulisai_xgb_model_retrained.joblib'
        joblib.dump(model, model_path)
        print(f"   Model saved to {model_path}")

        # Save feature names and label encoder
        joblib.dump(features, 'selected_features_retrained.joblib')
        joblib.dump(label_encoder, 'label_encoder_retrained.joblib')

        return True, model_path

    except Exception as e:
        print(f"   Error during model retraining: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

#Run the App
if __name__ == '__main__':
    app.run(debug=os.environ.get('PULISAI_DEBUG') == '1')