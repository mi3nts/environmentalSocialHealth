import pandas as pd
import os 
import xarray as xr
import geopandas as gpd
import numpy as np

root = "./CAMS/"
units = "./CAMS_units.csv"
zip_2010 = "./tx_texas_zip_codes_geo.min.json"
hospital_data = "/media/teamlary/ssd/Discharge Data/Inpatient/Data/"
census_data = "./Census/"

units_df = pd.read_csv(units)
tx_zip = gpd.read_file(zip_2010)

start_year = 2005
end_year = 2022

hospital_quarters = [f"{quarter}q{year}" for year in range(start_year, end_year + 1) for quarter in range(1, 5) if not (year == end_year and quarter > 2)]
icd9_subset = hospital_quarters[hospital_quarters.index(f'1q{start_year}'):hospital_quarters.index('4q2015')]
icd10_subset = hospital_quarters[hospital_quarters.index('4q2015'):hospital_quarters.index(f'1q{end_year}')]

list_of_quarters = (pd.date_range(pd.to_datetime(f'{start_year}-01-01'), 
                   pd.to_datetime('2022-03-31') + pd.offsets.QuarterBegin(1), freq='Q')
      .strftime('%Y-%m-%d')
      .tolist())
# list_of_quarters = ["1q2012","2q2012","3q2012","4q2012"]

icd9_subset_env = list_of_quarters[list_of_quarters.index(f'{start_year}-03-31'):list_of_quarters.index('2015-12-31')]
icd10_subset_env = list_of_quarters[list_of_quarters.index('2015-12-31'):list_of_quarters.index('2022-03-31')]


def grabWeatherData(time_period):
    means = pd.DataFrame()
    for quarter in time_period:
        '''
        Objective: create a dataframe of mean meterological values per zip code per quarter, over multiple quarters. 
        This way, we can merge multiple quarter data points with hospitalization data.
        In one for loop, we may have to load mean zip code data for quarter, hospitalization data for quarter, merge, then delete the loaded files, and repeat

        '''

        ### load mean weather data into means (pd.DataFrame)
        csvs = [root+quarter+'/'+i for i in os.listdir(root+quarter)]
        # num_csvs = len(csvs)
        for i in csvs:
            df = pd.read_csv(i, skiprows=0, usecols=lambda x: x != 'Unnamed: 0')
            means = pd.concat([means, df])
        
        # print(means.head())

        # means.insert(0, 'PAT_ZIP', tx_zip['ZCTA5CE10'].values)
    # means.columns = units_df['var_name'].values
    means.insert(0, 'PAT_ZIP', tx_zip['ZCTA5CE10'].to_list()*len(time_period))
    means.insert(1, 'LandArea_sqm', tx_zip['ALAND10'].to_list()*len(time_period))
    means['PAT_ZIP'] = means['PAT_ZIP'].astype('int')

    return means

icd9_env = grabWeatherData(icd9_subset_env)
icd10_env = grabWeatherData(icd10_subset_env)

env_data = pd.concat([icd9_env, icd10_env])
second_index = [i for i in icd9_subset + icd10_subset for _ in range(1939)]
env_data.insert(0, 'quarter', second_index)
col_names = ['quarter', 'PAT_ZIP', 'LandArea_sqm'] + list(units_df['var_name'].values[3:])
env_data.columns = col_names

env_data.to_csv("all_env_data_2005q1_2021q4.csv", index=False)