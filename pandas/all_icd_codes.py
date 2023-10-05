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

def interpCensus(quarter):
    year = quarter[2:]
    interpolatedPops = pd.read_csv("Census/interpolatedPopulations.csv")
    df = interpolatedPops[['PAT_ZIP',year]]
    df.columns = ['PAT_ZIP','population']

    return df
    
def loopICDperQuarter(time_period, nthresh=1, icd10=True):
    data = pd.DataFrame()
        ### load hospital data into hospitaldf (pd.DataFrame)
    for ind, quarter in enumerate(time_period):
        print(quarter)
        hospital_df = pd.read_csv(f"{hospital_data}PUDF_base1_{quarter}_tab.csv")
        # print('hospital df head')
        # print(hospital_df.head())

        hospital_df['PAT_ZIP'] = hospital_df['PAT_ZIP'].astype('str')
        hospital_df = hospital_df[~hospital_df['PAT_ZIP'].str.endswith('.0')]
        hospital_df = hospital_df[hospital_df['PAT_ZIP'].map(len) == 5]
        hospital_df['PAT_ZIP'] = hospital_df['PAT_ZIP'].astype('int')

        zip_population = interpCensus(quarter)

        if icd10 == True:
            codes = icd['ICD-10']
        else:
            codes = icd['PRINC_DIAG_CODE']

        for sind, icd_code in enumerate(codes[:2000]):
            os.makedirs(f'icdQuarter/{icd["PRINC_DIAG_CODE"][sind]}', exist_ok=True)
            icd_df = groupDF(hospital_df, icd_code)
            icd_df = icd_df.merge(zip_population, on='PAT_ZIP')
            icd_df['normalized'] = icd_df['ICD']/icd_df['population']
            # icd_df = icd_df[['PAT_ZIP','ICD','normalized']]
            icd_df = icd_df[icd_df['ICD'] >= nthresh]
            icd_df.to_csv(f'./icdQuarter/{icd["PRINC_DIAG_CODE"][sind]}/{quarter}.csv', index=False)
            # print(icd_df.head())

    del hospital_df

    return data