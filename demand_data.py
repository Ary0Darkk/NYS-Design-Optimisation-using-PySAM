import pandas as pd

df = pd.read_excel('electricity_data/Yearly_Demand_Profile_state_mahrastra_and_manipur.xlsx')

# print(df.head(5))
# print(type(df))

df_2024 = df[df['State'] == 'Maharashtra - 2024'].copy()

# print('Data for 2024')
# print(df_2024.head(5))

df_2024['DateTime'] = pd.to_datetime('2024-' + df_2024['Date'],
    format='%Y-%d-%b %I%p'
)

# Create a boolean mask where Month is 2 (February) AND Day is 29.
is_leap_day = (df_2024['DateTime'].dt.month == 2) & (df_2024['DateTime'].dt.day == 29)

# Use the NOT operator (~) to keep all rows EXCEPT those that are the leap day.
df_2024_no_leap = df_2024[~is_leap_day].copy()



# print(df_2024_no_leap.head(30))
# print(len(df_2024_no_leap))

formatted_demand_df = df_2024_no_leap[["DateTime","Hourly Demand Met (in MW)"]].reset_index(drop=True)

# print(formatted_demand_df.head())
# print(len(formatted_demand_df))