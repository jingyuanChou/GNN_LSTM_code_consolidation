import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import *
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

region='VA'
model_name = '../model/model_lstm-{}.h5'.format(region)

print('REGION:{}'.format(region))

print('Preparing ground truth.')
from datetime import datetime


def lstm_mcdropout(input_shape, hidden_rnn, hidden_dense, output_dim, activation):
    inp = Input(shape=input_shape)
    left = Sequential()
    left = LSTM(input_shape=input_shape,units=hidden_rnn,return_sequences=False)(inp)
    left = Dense(units=hidden_dense,activation=activation)(left)
    left = Dropout(0.2)(left, training=True)
    out = Dense(units=output_dim,activation='linear',kernel_regularizer=regularizers.l2(0.01))(left)
    return Model(inputs=inp, outputs=out, name='lstm_mcdropout')

def get_week(date, weeks):
    for week in weeks:
        s, e = week.split('_')
        if s <= date and date <= e:
            return week

def predict_n_point(model, X, steps, scaler, len_scal, nth_scal, n_back, n_feature=1):
    ### N step ahead forecasting (point estimate) ###
    points = []
    p = X
    for i in range(steps):
        pred = model.predict(p.reshape(-1,n_back,n_feature))
        tran_pd = np.asarray([pred[0]]*len_scal).reshape(1,len_scal)
        point = scaler.inverse_transform(tran_pd)
        points.append(point[0][nth_scal])
        p = np.append(p,pred[0])[1:]
    return points

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

if __name__ == '__main__':

    # Confirmed cases
    initial_sequence_length = 12
    file = "covid-data/weekly_filt_case_state_level_2020_2022.csv"
    new_df = pd.read_csv(file)
    dates = new_df.columns
    new_df = new_df.set_index('location')
    for dt in new_df.columns:
        new_df = new_df.rename(
            columns={dt: dt + '_' + (datetime.strptime(dt, '%m/%d/%Y') + timedelta(days=6)).strftime('%m/%d/%Y')})
    dates = new_df.columns.values.tolist()

    gt = new_df.copy()
    gt[gt < 0] = 0.0
    gt = gt.T
    # gt = gt[gt.index>=skip_date]
    gt_dates = gt.index.values
    gt = gt.astype('float')

    sgt = gt.copy()
    all_regions = gt.columns.tolist()
    regions = all_regions
    timestamps = np.arange(4, len(gt)-initial_sequence_length)

    selected_states = {51:'VA',45:'SC',37:'NC',13:'GA',47:'TN'}
    for k, s in enumerate(regions):
        for time in timestamps:
            print('Prepare training data...')
            train = sgt.iloc[0:initial_sequence_length + time][[s]] # 0:12+4
            scaler = MinMaxScaler()
            scaler.fit(train.values)
            train.values[:, :] = scaler.transform(train.values)

            n_back = 12
            n_ahead = 1
            n_feature = 1
            n_in = n_back * n_feature
            n_out = n_ahead * n_feature

            values = train[s].values.tolist()
            reframed = series_to_supervised(values, n_back, n_ahead)
            data = reframed.values
            train_X, train_Y = data[:, :n_in], data[:, -n_out:]
            # reshape inputs to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], n_back, n_feature))
            train_Y = train_Y.reshape(-1, n_ahead, n_feature)
            print('{} training data is prepared.'.format(s))

            print(train_X.shape, train_Y.shape)

            print('Building model...')
            model_type = 'lstm_mcdropout'
            input_shape = (n_back, n_feature)
            hidden_rnn = 32
            hidden_dense = 16
            output_dim = n_ahead
            activation = 'relu'
            model = lstm_mcdropout(input_shape, hidden_rnn, hidden_dense, output_dim, activation)
            model.compile(loss='mse', optimizer='adam', metrics=['mse'])
            # Model saving path
            filepath = model_name
            print('Preparing callbacks...')
            # Prepare callbacks for model saving and for learning rate adjustment.
            checkpoint = ModelCheckpoint(filepath=filepath,
                                         monitor='val_loss',
                                         verbose=0,
                                         save_best_only=True,
                                         mode='min')

            earlystop = EarlyStopping(monitor='val_loss',
                                      patience=50,
                                      verbose=0,
                                      mode='min',
                                      restore_best_weights=True)

            callbacks = [checkpoint, earlystop]

            print('Training...')
            batch_size = 32
            epochs = 200
            history = model.fit(train_X, train_Y,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                verbose=0,
                                shuffle=False,
                                callbacks=callbacks)

            print('Prepare testing data...')
            x = train[s].values[-n_back:].reshape((1, n_back, n_feature))

            print('Predicting...')
            horizon = 4
            mc_num = 50
            print('Predicting for {}...'.format(s))
            r, c = mc_num * horizon, 5
            predict_vec = np.zeros([r, c])
            nth_scal = 0
            len_scal = 1
            predict_vec[:r, 1:2] = np.array([0] * r).reshape(-1, 1)
            for mc in range(mc_num):
                pt_pd = predict_n_point(model, x, horizon, scaler, len_scal, nth_scal, n_back, n_feature)
                r3 = horizon
                predict_vec[mc * r3:(mc + 1) * r3, 2:3] = np.array(range(horizon)).reshape(-1, 1)
                predict_vec[mc * r3:(mc + 1) * r3, 3:4] = np.array([mc] * horizon).reshape(-1, 1)
                predict_vec[mc * r3:(mc + 1) * r3, 4:5] = np.array(pt_pd).reshape(-1, 1)
            if time == 4:
                pd_df = pd.DataFrame(predict_vec, columns=['fips', 'date', 'horizon', 'mc', '{}'.format(time+initial_sequence_length)])
                pd_df['fips'] = [selected_states[s]] * r
                if k == 0:
                    header = True
                else:
                    header = False
            else:
                pd_df['{}'.format(time+initial_sequence_length)] = predict_vec[:,-1]


        pd_df.to_csv('{}_case_dropout02_LSTM_only.csv'.format(selected_states[s]), header=header, index=False)
