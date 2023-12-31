import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import numpy as np
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot_2samples
import os 
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import re
import simple_icd_10_cm as cm
import textwrap

root = "../CAMS/"
units = "../assets/CAMS_units.csv"
zip_2010 = "../assets/tx_texas_zip_codes_geo.min.json"
hospital_data = "/media/teamlary/ssd/Discharge Data/Inpatient/Data/"
census_dir = "../Census/"
icd_data = "../icd10/"

pandas_or_polars = False

if pandas_or_polars:
    icd_data = "../icd10_pandas/"
    save_dir = 'pandas'
else:
    icd_data = '../icd10_polars/'
    save_dir = 'polars'

tx_zip = gpd.read_file(zip_2010)

start_year = 2005
end_year = 2022

hospital_quarters = [f"{year}q{quarter}" for year in range(start_year, end_year + 1) for quarter in range(1, 5) if not (year == end_year and quarter > 2)]
hospital_quarters = hospital_quarters[:-1]
icd9_subset = hospital_quarters[hospital_quarters.index(f'{start_year}q1'):hospital_quarters.index('2015q4')]
icd10_subset = hospital_quarters[hospital_quarters.index('2015q4'):hospital_quarters.index(f'{end_year}q1')]

def numerical_sort(filename):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', filename)]

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
        sorted_csvs = sorted(csvs, key=numerical_sort)
        # num_csvs = len(csvs)
        for i in sorted_csvs:
            df = pd.read_csv(i, skiprows=0, usecols=lambda x: x != 'Unnamed: 0')
            means = pd.concat([means, df])
        
        # print(means.head())

        # means.insert(0, 'PAT_ZIP', tx_zip['ZCTA5CE10'].values)
    # means.columns = units_df['var_name'].values
    means.insert(0, 'PAT_ZIP', tx_zip['ZCTA5CE10'].to_list()*len(time_period))
    means.insert(1, 'LandArea_sqm', tx_zip['ALAND10'].to_list()*len(time_period))
    means['PAT_ZIP'] = means['PAT_ZIP'].astype('int')

    return means

env_quarters = (pd.date_range(pd.to_datetime(f'{start_year}-01-01'), 
                   pd.to_datetime('2021-12-31') + pd.offsets.QuarterBegin(1), freq='Q')
      .strftime('%Y-%m-%d')
      .tolist()) # from 2005-03-31 to 2021-12-31

env_data = grabWeatherData(env_quarters)
second_index = [i for i in icd9_subset + icd10_subset for _ in range(1939)]
env_data.insert(0, 'quarter', second_index)

units_df = pd.read_csv(units)
unit_names = ['quarter', 'PAT_ZIP','LandArea_sqm'] + list(units_df['var_name'].values[3:])
env_data.columns = unit_names
env_data = env_data.reset_index(drop=True)

nice_names = dict(zip(env_data.columns[3:], units_df['long_name'][3:].values))
nice_names['go3'] = 'Ozone mass mixing ratio'
nice_names['ch4_c'] = 'Methane'
nice_names['pop_density'] = 'Population Density'
nice_names['pm10'] = 'Particulate matter 10um'
nice_names['pm2p5'] = 'Particulate matter 2.5um'

env_data = env_data.copy().dropna(axis=1, how='all')
env_data = env_data.dropna()

print("\n\n Environmental data processed. \n\n")

quarter_label = dict(zip(env_quarters, [i.upper() for i in icd9_subset + icd10_subset]))


def eval_results(y_test, predictions, y_train, train_preds):
    train_acc, test_acc = [], []
    mse = mean_squared_error(y_test, predictions)
    # print("RMSE: ", np.sqrt(mse))
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, predictions)
    train_acc.append(train_r2)
    test_acc.append(test_r2)
    # print("Train R2: ", train_r2)
    # print("Test R2: ", test_r2)
    #print(train_preds)

    # bin_lbl = bin_labels['labels'].loc[bin_labels['bin_id'] == item].values[0]

    train_pdf = pd.DataFrame.from_dict({'Predicted': train_preds, 'Actual': y_train, 'Legend': 'Train'})
    test_pdf = pd.DataFrame.from_dict({'Predicted': predictions, 'Actual': y_test, 'Legend': 'Test'})
    full_pdf = pd.concat([train_pdf, test_pdf])
    #print('len of full pdf inside fxn', len(full_pdf))
    
    return mse, train_r2, test_r2, train_acc, test_acc, full_pdf

def XGB_model(X_train, X_test, y_train, y_test):
    # dt = xgb.DMatrix(X_train, label=y_train.values)
    # dv = xgb.DMatrix(X_test, label=y_test.values)
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        # "eta": 0.05,
        # "max_depth": 10,
        # "n_estimators": 100,
        # "num_boost_round": 10,
        "objective": "reg:squarederror",
        "device": "cuda",
        # "verbose": 0,
        # "verbosity": 0,
        # "silent": 1,
        # "base_score": np.mean(y_test),
        "eval_metric": "rmse"
    }
    # model = xgb.train(params, dt, 300, [(dt, "train"),(dv, "valid")])#, early_stopping_rounds=10)
    model = xgb.train(
    params=params, 
    dtrain=dt, 
    num_boost_round=300, 
    evals=[(dt, "train"), (dv, "valid")],
    early_stopping_rounds=10,
    verbose_eval=False  # You can set verbose_eval to True or False as needed
)

    
    predictions = model.predict(dv)
    train_preds = model.predict(dt)
    
    return model, predictions, train_preds

def test_train_plot(full_pdf, y_test, train_r2, y_train, X_train, test_r2, X_test, title, save_path):
    plt.figure(figsize=(12, 8)) 
    print('inside r2 plot')
    #print('len of full pdf inside plot fxn', len(full_pdf)) 
    
    g = sns.jointplot(data=full_pdf, x='Predicted', y='Actual', hue='Legend', alpha=0.3, #cmap="Blues", 
                          marginal_kws=dict(bw_adjust=0.2, cut=0))
    g.set_axis_labels('Predicted ($\log_{10}$(# ICD Codes/Zip Code Population))', 'Actual ($\log_{10}$(# ICD Codes/Zip Code Population))')
    set_min_plot = y_train
    g.ax_joint.plot([min(set_min_plot), max(set_min_plot)], [min(set_min_plot), max(set_min_plot)], color='k',label="1:1")
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=[ f'Train: {train_r2:.2f}, N = {len(X_train):,}', f'Test: {test_r2:.2f}, N={len(X_test):,}','1:1',], title=None)
    g.fig.suptitle(title, size=20)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    # plt.show()

    
def plot_qq(full_pdf, save_path, title):
    print('inside qq plot')
    fig = qqplot_2samples(full_pdf['Predicted'],full_pdf['Actual'],line='45', 
                    xlabel='Predicted ($\log_{10}$(# ICD Codes/Zip Code Population))',  
                    ylabel='Actual ($\log_{10}$(# ICD Codes/Zip Code Population))')
    plt.title(f'{title}') 
    # plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

def shap_plots(model, X_test, save_path, title):
    print('inside shap plot')
    explainer = shap.TreeExplainer(model)

    # feature_names = [nice_names[i] for i in X_test.columns]
    # print(feature_names)
    # feature_names = X_test.columns
    feature_names = {i: nice_names[i] for i in X_test.columns}
    # print([(i,j,nice_names[j]) for i,j in  enumerate(feature_names)])
    # print(feature_names)
    # X_test = pd.DataFrame(X_test)
    X_test.columns = [i if i not in feature_names.keys() else feature_names[i] for i in feature_names]
    # print(X_test.head)
    shap_values = explainer.shap_values(X_test)
    
    # plt.figure(figsize=(18,12))
    shap.summary_plot(shap_values, X_test, plot_type='dot', max_display=10, show=False, plot_size=[16,8])
    # print('part2')
    # fig, ax = plt.gcf(), plt.gca()
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    # print(feature_order, X_test.columns[feature_order])
    # ax.set_yticklabels([corr_labels[i] for i in selected_columns])
    # ax.set_yticklabels([selected_columns[i] for i in feature_order])

    # ax.set_yticklabels([corr_labels[i] for i in features.columns[feature_order]])
    plt.xlabel('Average Impact on Model')
    # plt.title(f'Feature Importance Ranking for {plot_title}{pm_labels[item]}')
    # print(f"Shap summary for {item}")
    plt.title(f'{title}') 
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

icd_codes = ['I5042', 'A4189', 'I509', 'I2510','F209','G309', 'J4521']
icd_codes = os.listdir('../icd10/')

def getDF(icd_codes):
    fits_data = []
    nthresh = 3
    data_list = []
    for icd_code in icd_codes:
        df = pd.DataFrame()
        for quarter in hospital_quarters[:-1]: # [:-1] to avoid 2022q1 

            # read in icd data
            icd_df = pd.read_csv(f'{icd_data+icd_code}/{quarter}.csv') 
            
            # environmental data is already read in
            # for each quarter, .loc env_data that is on that quarter, then 
            # merge with zip codes that are in icd_df
            # this merged df needs to be concat into df for every quarter
            env_df = env_data[env_data['quarter'] == quarter]
            
            # census_df = census_data[census_data['year'] == quarter[:4]]
            # env_icd = env_df.merge(icd_df, on='PAT_ZIP')

            # full_df = census_df.merge(env_icd, on='PAT_ZIP')
            full_df = env_df.merge(icd_df, on='PAT_ZIP')
            full_df['pop_density'] = full_df['population']/(full_df['LandArea_sqm']/1_000_000)
        

            df = pd.concat([df, full_df])
        
        os.makedirs(f'./Plots/{icd_code}', exist_ok=True)

        data = df.copy()
        data_quality = data.reset_index(drop=True)
        # data_quality = data_quality.drop(census_data.columns[2:], axis=1) # dropping all census data
        # print(len(data_quality))
        data_quality = data_quality[data_quality['ICD'] >= nthresh]
        #print(f'Len of data at no threshold: {len(data)}, len of data at threshold: {len(data_quality)}')
        
        # data_quality = data_quality.drop([data_quality.columns[data_quality.isna().any()]][0], axis=1)
        # print(len(data_quality))
        # data_quality = np.log10(data_quality)
        
        data_quality.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_quality = data_quality.dropna(axis=0)
        data_list.append(data_quality)
        # print(data_quality.columns)
        # print(len(data_quality))
        # data_quality = data_quality[data_quality['median household income'] > 0]
        # X = data_quality.drop(['PAT_ZIP','ICD','normalized', 'population', 'quarter'], axis=1)
        # X = data_quality.drop(['LandArea_sqm'], axis=1) # dropping land area in sqm
        # print(len(X))

        # "chnk", \
        X = data_quality.loc[:,["d2m","t2m", "lai_hv","lai_lv", 
        "pm10","pm2p5","stl1",
        "sp",
        "co", "aermr04","aermr05","aermr06", 
        "c2h6","hcho","aermr09","aermr07","aermr10",
        "aermr08","oh", "c5h8", "ch4_c",
        "hno3","no2","no","go3","pan",
        "c3h8", "aermr01","aermr02","aermr03",
        "aermr12",
        "aermr11",
        "so2"]] #, "pop_density"]] # including only relevant environmental data
        # "median household income", \
        # "hispanic",
        # # "aggregate income",\
        # "males college < 1yr", \
        # "males college > 1yr, no degree","males associate degree", "SNAP eligibility",]]
        # X = X.dropna()
        # print(X.shape)

        # data_quality['normalized'].plot()

        # return df
        sns.set_style('ticks')
        # rmse_list = []
        # for i in range(500):
        y = np.log10(data_quality['normalized'])
        # y = data_quality['normalized']
        # y = data_quality['ICD']
        # seed = 140

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=seed)

        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
        y_test_scaled = target_scaler.transform(y_test.to_numpy().reshape(-1,1))

        
        # model = GradientBoostingRegressor()

        # model = ExtraTreesRegressor()

        # Train and predict using each model
        
        # for model_name, model in models.items():
        # print('X_train', len(X_train_scaled), ' y_test', len(y_test))
        # model.fit(X_train_scaled, y_train)
        # y_pred = model.predict(X_test_scaled)

        # r2_scores = r2_score(y_test, y_pred)
        # print(r2_scores)
        # rmse_list.append(r2_scores)

        # -----------

        # Get feature importances

        # things that worked

        # model = RandomForestRegressor()
        # predictions = {}
        # model.fit(X_train_scaled, y_train)
        icd_code_title = icd_code
        # y_pred = model.predict(X_test_scaled)
        # r2_scores = r2_score(y_test, y_pred)
        # train_preds = model.predict(X_train_scaled)

        model, y_pred, train_preds = XGB_model(X_train, X_test, 
                                               y_train, y_test)


        mse, train_r2, test_r2, train_acc, test_acc, full_pdf = eval_results(y_test, y_pred, y_train, train_preds)
        # print('full pdf', len(full_pdf))
        # print('sus error start')
    
        test_train_plot(full_pdf, y_test, train_r2, y_train, X_train_scaled, test_r2, X_test_scaled,
                        title=textwrap.fill(f"ICD-10 Codes for {icd_code_title} \n # threshold = {nthresh}, \
                        Environmental data \n from {start_year} to {end_year-1}"),
                        save_path=f'../Plots_{save_dir}/{icd_code}/{icd_code}_r2.png'
                        )
        # print('sus error end')

        if cm.is_valid_item(icd_code):
            icd_code_title = cm.get_description(icd_code)
        else: 
            icd_code_title = icd_code
        
        model_rf = RandomForestRegressor()
        model_rf.fit(X_train_scaled, y_train)

        importance_scores = model_rf.feature_importances_
        feature_names = [nice_names[i] for i in X.columns]
        # feature_names = X.columns

        # Sort feature importances in descending order
        indices = importance_scores.argsort()[::-1][:20]
        sorted_feature_names = ([feature_names[i] for i in indices])
        sorted_importance_scores = (importance_scores[indices])

        # Create a horizontal bar chart of feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importance_scores)), sorted_importance_scores[::-1], align='center')
        plt.yticks(range(len(sorted_importance_scores))[::-1], sorted_feature_names)

        plt.title(textwrap.fill(f'Feature Importance Ranking for Environmental data model on {icd_code_title}'))
        plt.ylabel('Features')
        plt.xlabel('Feature Importance Ranking')
        plt.savefig(f'../Plots_{save_dir}/{icd_code}/{icd_code}_feat_imp.png', bbox_inches='tight')
        plt.clf()

        shap_plots(model, 
            X_test, 
            f'../Plots_{save_dir}/{icd_code}/{icd_code}_shap.png', 
            title=textwrap.fill(f"SHAP Values for {icd_code_title}"))


        plot_qq(full_pdf, f'../Plots_{save_dir}/{icd_code}/{icd_code}_qq.png')

        del full_pdf, df

        fits_data.append([icd_code, train_r2, test_r2, np.sqrt(mse), len(X)])

    fits_df = pd.DataFrame(fits_data)
    fits_df.columns = ['ICD', 'train_r2', 'test_r2', 'rmse', 'numDataPoints']

    return data_list


    # fits_df.to_csv(f'n_thresh = {nthresh} - results.csv')

icd_code_labels = {
    'A4189': 'A41.89 - Other Specified Sepsis',
    'B9789': 'B97.89 - Other viral agents as the cause \n of diseases classified elsewhere',
    'J157' : 'J15.7 - Pneumonia due to Mycoplasma pneumoniae',
    'J449' : 'J44.9 - Chronic obstructive pulmonary disease, unspecified',
    'J45998' : 'J45.998 - Other asthma',
    'J4521' : 'J45.21 - Mild intermittent asthma \n with (acute) exacerbation', 
    'F068' : 'F06.8 - Other specified mental disorders \n due to known physiological condition', 
    'I2510' : 'I25.10 - Atherosclerotic heart disease \n of native coronary artery without angina pectoris',
    'I509' : 'I50.9 - Heart failure, unspecified', 
    'A4189' : 'A41.89 - Other specified sepsis',
    'I5041' : 'I50.41 - Acute combined systolic (congestive) \n and diastolic (congestive) heart failure',
    'I5042' : 'I50.42 - Chronic combined systolic (congestive) \n diastolic (congestive) heart failure',
    'F209' : 'Schizophrenia, unspecified'
}

#icd_codes = ['A4189', 'I5042','I509', 'I2510','F209','G309', 'J4521']
#icd_codes_subset = list(icd_code_labels.keys())[:1]
icd_codes_subset = ['A419','I2510','E860','J189'][:1]

print("Creating Models... \n\n")
data_for_icd = getDF(icd_codes_subset)
data_for_icd[0].to_csv(f'{icd_codes_subset[:1]}.csv',index=False)

