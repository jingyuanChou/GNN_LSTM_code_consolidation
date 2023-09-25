import json
import pandas as pd
import numpy as np
import datetime

def main():
    reader = open('5_states_weekly_cases_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json')
    reader2 = open('Cases_5_states_weekly_recrusive_predictions_vs_labels_4_week_ahead_regularized_12_weeks_mean_variance.json')
    one_week = json.load(reader)
    LSTM_ONLY = json.load(reader2)
    # '1 wk ahead inc case', '2 wk ahead inc case', '3 wk ahead inc case', '4 wk ahead inc case'
    # so first forcast date in the result is 2020-10-25, and we just add 7 more days on top of that
    quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    fips = {'VA':51,'SC':45,'NC':37,'GA':13,'TN':47}
    start_time = datetime.datetime.strptime('2020-09-27','%Y-%m-%d').date()
    for index2 in range(4,71):
        total_result = list()
        for key in fips.keys():
            result = LSTM_ONLY[key]
            predictions = result['recursive_next_4_prediction']
            labels = one_week[key]['labels']
            if key == 'VA':
                another_time = start_time + datetime.timedelta(days=(index2-4) * 7)
                forecast_date = another_time + datetime.timedelta(days=1)
                next_1_week = forecast_date + datetime.timedelta(days=5)
                next_2_week = forecast_date + datetime.timedelta(days=12)
                next_3_week = forecast_date + datetime.timedelta(days=19)
                next_4_week = forecast_date + datetime.timedelta(days=26)
            for idx, each_timestamp in enumerate(predictions[str(index2)]):
                label_each = labels[index2-4]
                for quant in quantiles:
                    quantile_case = np.quantile(each_timestamp, quant)
                    if idx == 0:
                        target_date = next_1_week
                    elif idx == 1:
                        target_date = next_2_week
                    elif idx == 2:
                        target_date = next_3_week
                    else:
                        target_date = next_4_week
                    temp = [forecast_date, '{} wk ahead inc case'.format(str(idx+1)), target_date, fips[key], 'quantile', quant, quantile_case, label_each[idx]]
                    total_result.append(temp)
                    point_ls = [forecast_date, '{} wk ahead inc case'.format(str(idx+1)), target_date, fips[key], 'point', np.nan, np.median(each_timestamp), label_each[idx]]
                    total_result.append(point_ls)

        total_result = pd.DataFrame(total_result)
        total_result.drop_duplicates(inplace=True)
        total_result.fillna('NA', inplace=True)
        total_result.columns = ['forecast_date', 'target', 'target_end_date', 'location', 'type', 'quantile', 'value','truth']
        total_result.to_csv('ONLY_LSTM_result_{}.csv'.format(forecast_date))

if __name__ == '__main__':
    main()



