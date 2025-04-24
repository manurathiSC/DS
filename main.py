# -*- coding: utf-8 -*-
"""Data Processing.ipynb
"""



import awswrangler as wr
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import boto3
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import config 

def getCols(tableName):
    schema = wr.catalog.get_table_types(database='prod_da', table=tableName)
    cols = [key if val!='timestamp' else f"cast({key} as timestamp) as {key}" for key, val in schema.items()]
    return cols

def getTableData(subTableName):
    cols = getCols(subTableName)
    mainDf = wr.athena.read_sql_query(sql=f"select {', '.join(cols)} from {subTableName}",
                              database="prod_da",
                              workgroup='data_analytics',)
    return mainDf

def getSubData(mainDf, nonSubsDf):
    mainDf['Days_ECO'] = pd.Series(mainDf['first_sub_date'] - mainDf['first_activity_date'],index = mainDf.index)
    mainDf['Days_ECO'] = [c.days for c in mainDf['Days_ECO']]
    mainDf = mainDf[mainDf['first_sub_date'].notna()]

    # nonSubsDf = nonSubsDf.set_index('sam_id').reindex(columns=nonSubsDf.set_index('sam_id').columns.union(mainDf.set_index('sam_id').columns))
    mainDf = mainDf.reindex(columns=nonSubsDf.columns)
    nonSubsDf.set_index('sam_id').update(mainDf.set_index('sam_id'))

    nonSubsDf.reset_index(inplace=True)
    return nonSubsDf

def get_table(table_name, table_date, samIDsr = False, anchor_date = None, mainDf = None, nonSubsDf = None):
    cols1 = getCols(table_name)
    if nonSubsDf is None:
        nonSubsDf = wr.athena.read_sql_query(sql=f"select {', '.join(cols1)} from {table_name}",
                              database="prod_da",
                              workgroup='data_analytics',)

    ffilter = pd.Series([firstCond or secondCond for firstCond, secondCond in
                         zip(nonSubsDf['first_sub_date'] > pd.to_datetime(table_date),
                                    nonSubsDf['first_sub_date'].isna())], index = nonSubsDf.index)

    nonSubsDf = nonSubsDf[ffilter]

    nonSubsDf = nonSubsDf[nonSubsDf['first_activity_date'] < pd.to_datetime(table_date)]
    if anchor_date is not None: nonSubsDf = nonSubsDf[nonSubsDf['first_activity_date'] > pd.to_datetime(anchor_date)]
    nonSubsDf['Days_ECO'] = pd.Series(pd.to_datetime(table_date) - nonSubsDf['first_activity_date'],index = nonSubsDf.index)
    nonSubsDf['Days_ECO'] = [c.days for c in nonSubsDf['Days_ECO']]


    ysr = nonSubsDf['first_sub_date'].notna().astype('int')
    convSr = [c.days for c in (nonSubsDf['first_sub_date']- pd.to_datetime(table_date))]
    convSr = pd.Series(convSr, index = nonSubsDf.index)

    if samIDsr: samids = nonSubsDf['sam_id']

    if mainDf is not None: nonSubsDf = getSubData(mainDf,nonSubsDf)

    nonSubsDf = nonSubsDf[list(config.combined_set)]
    nonSubsDf[config.to_num] = nonSubsDf[config.to_num].astype(float)

    for col in config.to_num:
        if nonSubsDf[col].min()<0:
            nonSubsDf[col] = np.where(nonSubsDf[col]<0, 0 ,nonSubsDf[col])

    for col in config.to_num: nonSubsDf.loc[:,col] = np.clip(nonSubsDf[col],None,nonSubsDf[col].quantile(0.95))
    for col in config.to_num: nonSubsDf.loc[:,col] = StandardScaler().fit_transform(nonSubsDf[[col]])
    for col in config.to_num: nonSubsDf.loc[:,col] = nonSubsDf[col].fillna(nonSubsDf[col].min()-1)



    for bin_var in config.to_Bin: 
        nonSubsDf[bin_var] = nonSubsDf[bin_var].apply(lambda x: int(x) if pd.notnull(x) else -1)

    for prsnt_var in config.to_Prsnt:
        nonSubsDf[prsnt_var] = nonSubsDf[prsnt_var].notna().astype('int')

    # nonSubsDf['last_private'] = nonSubsDf['last_private'].apply(lambda x: int(x) if pd.notnull(x) else -1)
    # nonSubsDf['last_watchlist_private']  = nonSubsDf['last_watchlist_private'].apply(lambda x: int(x) if pd.notnull(x) else -1)
    # nonSubsDf['phone_filtered_id'] = nonSubsDf['phone_filtered_id'].notna().astype('int')
    # nonSubsDf['phone_watchlisted_id'] = nonSubsDf['phone_watchlisted_id'].notna().astype('int')


    # for dummy variable : form another dummies
    changeSubsDf = pd.get_dummies(nonSubsDf[config.to_Dummy], dtype=int)
    nonSubsDf = nonSubsDf.drop(columns = config.to_Dummy)
    nonSubsDf = nonSubsDf.join(changeSubsDf)

    for var_name in config.to_freq:
        tmp_fe = nonSubsDf.groupby(var_name).size()/len(nonSubsDf)
        nonSubsDf.loc[:,var_name + '_Freq'] = nonSubsDf[var_name].map(tmp_fe)
        nonSubsDf.loc[:,var_name + '_Freq'] = StandardScaler().fit_transform(nonSubsDf[[var_name + '_Freq']])
        nonSubsDf.loc[:,var_name + '_Freq'] = nonSubsDf[var_name + '_Freq'].fillna(nonSubsDf[var_name + '_Freq'].min()-1)

    nonSubsDf = nonSubsDf.drop(columns = config.to_freq)
    nonSubsDf = nonSubsDf.astype(float)

    if samIDsr:
        return nonSubsDf, ysr, convSr, samids
    else:
        return nonSubsDf, ysr, convSr

def trainModel(table_name, table_date, subDf, nonSubDf, s3_path):

    nonSubsDf,ysr,convSr = get_table(table_name, table_date, mainDf = subDf, nonSubsDf= nonSubDf)

    print("--Table reading Done--")

    rf = RandomForestClassifier(
        n_estimators = 1500, min_samples_leaf=500, n_jobs = -1, oob_score = True, random_state = 42,class_weight= "balanced")
    rf.fit(nonSubsDf, ysr)

    print("--Model Fit Done--")

    oob = rf.oob_score_

    buffer = io.BytesIO()
    joblib.dump(rf, buffer)
    buffer.seek(0)

    wr.s3.upload(local_file=buffer, path=s3_path)
    print("--Model Save Done--")

def getMetadataTable(tableName):
    year = int(tableName.split('_')[-3])
    month = int(tableName.split('_')[-2])
    day = int(tableName.split('_')[-1])

    modelPrefix = tableName.split('_')[0]
    
    return date(year, month, day), str(year) + "_"+str(month) + "_"+str(day), modelPrefix[2:]




# --------- For prediction -------    


def getModel(s3_path):
    buffer = io.BytesIO()
    wr.s3.download(path=s3_path, local_file=buffer)
    buffer.seek(0) 
    rf = joblib.load(buffer)
    return rf
    
def storeOutput(df,outputs3csvPath):
    wr.s3.to_csv(
    df= df,
    path=outputs3csvPath,)
    print('output loade in s3')

def getProbScore(model,table_name_g, table_date_g, table_date, nonSubDf):
    nonSubsDf_g,ysr_g,convSr_g, samID_g = get_table(table_name_g, table_date_g,samIDsr = True,    anchor_date=table_date,nonSubsDf=nonSubDf)
    nonSubsDf_g = setSameCols(list(model.feature_names_in_), nonSubsDf_g)
    nonSubsDf_g = nonSubsDf_g.reindex(columns=list(model.feature_names_in_))
    resultDf = getResultSamDf(model, nonSubsDf_g,ysr_g, convSr_g, samID_g, str(table_date), subs= False)
    return resultDf

def getProbTable(s3Path, nonSubTableName, nonSubTableDate, modelDict):  
    #modelDict -> tableDate : model of thatv date
    nonSubDf = getTableData(nonSubTableName)
    print('Table Reading Done')
    allResultDf = pd.DataFrame()
    for tableDate in modelDict.keys():
        modelPath = modelDict.get(tableDate)
        model = getModel(modelPath)
        print('Model Reading Done')
        resultDf = getProbScore(model,nonSubTableName, nonSubTableDate,tableDate ,nonSubDf)
        allResultDf = pd.concat([allResultDf, resultDf], axis = 1)
        print(tableDate)
    return allResultDf


def getProbSubDf(allResultDf, cutoff= None):
    if cutoff is None:
        probability_cols = [col for col in allResultDf.columns if col.startswith("probability")]
        sub_cols = [col for col in allResultDf.columns if col.startswith("Sub")]
    else:
        probability_cols = [
        col for col in allResultDf.columns
        if col.startswith("probability_") and pd.to_datetime(col.split("_")[1]) <= cutoff
        ]
        sub_cols = [
        col for col in allResultDf.columns
        if col.startswith("Sub_") and pd.to_datetime(col.split("_")[1]) <= cutoff
        ]
    probDf = allResultDf[probability_cols]
    subDf = allResultDf[sub_cols]
    return probDf, subDf

    
    probDf = allResultDf[probability_cols]
    subDf = allResultDf[sub_cols]
    return probDf, subDf

def setSameCols(nonSubsDfCols, nonSubsDf_g):
    colsToRemove = set(nonSubsDf_g.columns) - set(nonSubsDfCols)
    colsToAdd = set(nonSubsDfCols) - set(nonSubsDf_g.columns)

    if(len(colsToRemove) !=0): nonSubsDf_g = nonSubsDf_g.drop(columns = list(colsToRemove))
    if(len(colsToAdd) !=0):
        for col in colsToAdd:
            nonSubsDf_g.loc[:,col] = -1
    return nonSubsDf_g

def getFilteredDf(nonSubsDf,ysr):

    trn_df_1 = nonSubsDf[nonSubsDf['Days_ECO']>0]
    trn_y_1 = ysr.loc[trn_df_1.index]

    trn_df_0 = nonSubsDf[nonSubsDf['Days_ECO']<0]
    trn_y_0 = ysr.loc[trn_df_0.index]

    return trn_df_1,trn_y_1, trn_df_0, trn_y_0

def getResultDf(rf, nonSubsDf_g,ysr_g, convSr_g, table_date_g):
    prob = rf.predict_proba(nonSubsDf_g)
    tmpDf = pd.DataFrame(index =nonSubsDf_g.index ,columns = ['probability' ,'binary', 'decile','Days_ECO','Days_Conv'])
    tmpDf['probability'] = prob[:,-1]
    tmpDf['binary'] = ysr_g
    tmpDf['decile'] = [np.round(x) for x in tmpDf['probability']*10]
    tmpDf['Days_ECO'] = nonSubsDf_g['Days_ECO']
    tmpDf['Days_Conv'] = convSr_g

    grouped = tmpDf.groupby('decile')['binary']
    resultDf = pd.DataFrame(grouped.sum()/ysr_g.sum()).rename(columns={'binary': str(table_date_g) + '_Base'})
    resultDf[str(table_date_g) + '_Conv'] = grouped.sum()/grouped.count()
    resultDf[str(table_date_g) + '_Count'] = grouped.count()
    resultDf[str(table_date_g) + '_Subs'] = grouped.sum()

    resultDf[str(table_date_g) + '_EcoAvg'] = tmpDf.groupby('decile')['Days_ECO'].mean()
    resultDf[str(table_date_g) + '_SubsEcoAvg'] = tmpDf[tmpDf['binary'] == 1].groupby('decile')['Days_ECO'].mean()
    resultDf[str(table_date_g) + '_SubsConvAvg'] = tmpDf[tmpDf['binary'] == 1].groupby('decile')['Days_Conv'].mean()

    return resultDf

def getResultSamDf(rf, nonSubsDf_g,ysr_g, convSr_g, samid_g, tableName, subs= True):
    if subs:
        subIndex = ysr_g[ysr_g == 1].index
    else:
        subIndex = ysr_g.index
    prob = rf.predict_proba(nonSubsDf_g.loc[subIndex])
    idx_class_1 = list(rf.classes_).index(1)
    tmpDf = pd.DataFrame(index = subIndex)
    tmpDf['sam'] = samid_g.loc[subIndex]
    # tmpDf['probability_'+ tableName] = prob[:,-1]
    tmpDf['probability_'+ tableName] = prob[:,idx_class_1]
    tmpDf['conv_' + tableName] = convSr_g.loc[subIndex]
    tmpDf['Eco_' + tableName] = nonSubsDf_g.loc[subIndex]['Days_ECO']
    tmpDf['Sub_' + tableName] = ysr_g.loc[subIndex]
    tmpDf.set_index('sam', inplace = True)
    return tmpDf

def custom_convert(number):
    base = int(number * 10)  # Get the base part (e.g., 5 from 0.55, 5 from 0.53)
    remainder = number - base / 10.0  # Get the decimal part (e.g., 0.05 from 0.55, 0.03 from 0.53)
    if remainder >= 0.025 and remainder < 0.075:
        return base + 0.5
    elif remainder >= 0.075:
        return base + 1
    else:
        return base


def getResultTPTdf(probDf,subDf, table_date_g, stats, fiveDecile= False):   
    
    tmpDf = pd.DataFrame(index =probDf.index ,columns = ['probability' ,'binary', 'decile'])
    if stats == "Max":
        tmpDf['probability'] = probDf.max(1)
    elif stats == "Avg":
        tmpDf['probability'] = probDf.mean(1)
    elif stats == "Min":
        tmpDf['probability'] = probDf.min(1)
    elif stats == "Latest":
        tmpDf['probability'] = probDf.ffill(axis=1).iloc[:, -1]
            
        
    tmpDf['binary'] = subDf.max(1)
    if fiveDecile:
        tmpDf['decile'] = [custom_convert(x) for x in tmpDf['probability']]
    else:
         tmpDf['decile'] = [np.round(x) for x in tmpDf['probability']*10]
    
    grouped = tmpDf.groupby('decile')['binary']
    resultDf = pd.DataFrame(grouped.sum()/subDf.max(1).sum()).rename(columns={'binary': str(table_date_g) + '_Base'})
    resultDf[str(table_date_g) + '_Conv'] = grouped.sum()/grouped.count()
    resultDf[str(table_date_g) + '_Count%'] = grouped.count() / grouped.count().sum()
    resultDf[str(table_date_g) + '_Count'] = grouped.count()
    resultDf[str(table_date_g) + '_Subs'] = grouped.sum()
    return resultDf

def getResultS3(probDf,subDf, stats, fiveDecile= False):   
    
    tmpDf = pd.DataFrame(index =probDf.index ,columns = ['probability' ,'binary', 'decile'])
    if stats == "Max":
        tmpDf['probability'] = probDf.max(1)
    elif stats == "Avg":
        tmpDf['probability'] = probDf.mean(1)
    elif stats == "Min":
        tmpDf['probability'] = probDf.min(1)
    elif stats == "Latest":
        tmpDf['probability'] = probDf.ffill(axis=1).iloc[:, -1]
            
        
    tmpDf['binary'] = subDf.max(1)
    if fiveDecile:
        tmpDf['decile'] = [custom_convert(x) for x in tmpDf['probability']]
    else:
         tmpDf['decile'] = [np.round(x) for x in tmpDf['probability']*10]

    resultDf = tmpDf[tmpDf['binary'] == 0]
    resultDf = resultDf.drop(columns = ['binary'])    
    return resultDf