import pandas as pd
import numpy as np
import os,sys
from functools import reduce

def get_WIS(df):
    '''Computes the WIS score given a forecast dataframe in the standard ForecastHub quantile submission format. This code uses the eq. (4) of Evaluating epidemic forecasts in an interval format'''
    df=df[df['type']!='point']
    df.loc[:,'QS']=(2*((df.truth<=df.value)-df['quantile'])*(df['value']-df.truth))
    wdf=df.groupby(['location','forecast_date','target_end_date','target'],as_index=None).mean().rename(columns={'QS':'WIS'}).drop(['quantile','value'],axis=1)
    return wdf

def get_AE(df):
    '''Computes the AE score given a forecast dataframe in the standard ForecastHub quantile submission format with point forecasts. Note Flusight baseline does not have a point forecast.'''
    mdf=df[(df['type']=='point')]
    mdf.loc[:,'MAE']=np.abs(mdf['value']-mdf['truth'])
    mdf=mdf.rename(columns={'value':'point_fct'}).drop(['type','quantile'],axis=1)
    return mdf

def get_cov(df,cov=0.95):
    '''Produces the binary coverage given a forecast dataframe in the standard ForecastHub quantile submission format. 
    cov: Coverage values. Should be float. Default coverage is 0.95 and checks if ground truth lies between 0.025 and 0.975 quantile values.'''
    qk=round((1-cov)/2,3)    
    qdf=df[(df['quantile'].isin([qk,1-qk]))]
    qdf=pd.pivot_table(qdf,values='value',index=['location','forecast_date','target_end_date','target','truth'],columns=['quantile']).reset_index()
    qdf.loc[:,'COV_{}'.format(int(cov*100))]=((qdf['truth']>=qdf[qk]) & (qdf['truth']<=qdf[1-qk])).apply(int)
    return qdf

def get_eval(df):
    '''Produces the multi-score dataframe given a forecast dataframe in the standard ForecastHub quantile submission format.'''
    df1=get_WIS(df)
    df2=get_AE(df)
    df3=get_cov(df,cov=0.95)
    df4=get_cov(df,cov=0.5)
    dfs = [df1, df2, df3, df4]
    nedfs = [df for df in dfs if not df.empty]
    alldf = reduce(lambda left,right: pd.merge(left,right), nedfs)
    return alldf

if __name__ == '__main__':
    locations = ['51','45','37','13','47']
    weeks = ['1 wk ahead inc case','2 wk ahead inc case','3 wk ahead inc case','4 wk ahead inc case']
    quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    df = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/result_evaluation/GraphLSTM_result_2020-10-05.csv')
    df = df.iloc[:,1:]
    label = pd.read_csv('/Users/bijiehao/Downloads/causal_gnn_KDD/result_evaluation/2020-10-05-COVIDhub-ensemble.csv')
    label = label[label['location'].isin(locations)]
    label = label[label['target'].isin(weeks)]
    label.reset_index(inplace=True)
    label = label.iloc[:,1:]
    label['truth'] = 0

    for i in range(len(label)):
        td = label.iloc[i]['target_end_date']
        loc = label.iloc[i]['location']
        sub1 = df[df['target_end_date'] == td]
        sub2 = sub1[sub1['location'] == int(loc)]
        truth_value = sub2['truth'].values[0]
        label.iloc[i,7] = truth_value

    df = df.sort_values(['target_end_date', 'location','quantile'], ascending=[True, False, False])
    label = label.sort_values(['target_end_date', 'location','quantile'], ascending=[True, False, False])

    WIS_mine = get_WIS(df)
    WIS_ensemble = get_WIS(label)


    print('fine')
