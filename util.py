import pandas as pd
import numpy as np
import os

def generate_npy():
    df_id = pd.read_csv('data/city_info.csv')
    ids = df_id['ID'].unique()
    X_temp_train = None
    X_coordinate_train = None
    Y_temp_train = None
    X_temp_test = None
    X_coordinate_test = None
    Y_temp_test = None
    for city_id in ids:
        print(f'Processing {city_id}')
        df_city = pd.read_csv(os.path.join('data', f'{city_id}.csv'))
        df_city['tavg'] = (df_city['tmax'] + df_city['tmin']) / 2
        df_city['Date'] = pd.to_datetime(df_city['Date'])
        df_city = df_city[['Date', 'tavg']].dropna(axis=0)
        numy = int(len(df_city) / 365)
        numTrain = int(numy * 2 / 3)
        city = df_id[df_id['ID'] == city_id].iloc[0]
        coordinate = np.array([city['Lat'], city['Lon']])
        coordinate = coordinate.reshape(1, -1)
        coordinate /= 180
        for i in range(0, numy - 1, 2):
            X = (df_city['tavg'].iloc[365 * i:365 * (i + 1)] - 60) / 10
            Y = (df_city['tavg'].iloc[365 * (i + 1):365 * (i + 2)] - 60) / 10
            X_idx_diff = X.index.to_series().diff()
            Y_idx_diff = Y.index.to_series().diff()
            # skip time period if length of consecutive nan is more than 10 days
            if len(X_idx_diff[X_idx_diff > 10]) > 0 or len(Y_idx_diff[Y_idx_diff > 10]) > 0:
                continue
            X = np.array(X)
            X = X.reshape(1, -1, 1)
            Y = np.array(Y)
            Y = Y.reshape(1, -1, 1)
            if i < numTrain:
                if X_temp_train is None:
                    X_temp_train = X
                    Y_temp_train = Y
                    X_coordinate_train = coordinate
                else:
                    X_temp_train = np.concatenate((X_temp_train, X), axis=0)
                    Y_temp_train = np.concatenate((Y_temp_train, Y), axis=0)
                    X_coordinate_train = np.concatenate((X_coordinate_train, coordinate), axis=0)
            else:
                if X_temp_test is None:
                        X_temp_test = X
                        Y_temp_test = Y
                        X_coordinate_test = coordinate
                else:
                    X_temp_test = np.concatenate((X_temp_test, X), axis=0)
                    Y_temp_test = np.concatenate((Y_temp_test, Y), axis=0)
                    X_coordinate_test = np.concatenate((X_coordinate_test, coordinate), axis=0)

    np.save('data/X_temp_train.npy', X_temp_train)
    np.save('data/X_coordinate_train.npy', X_coordinate_train)
    np.save('data/Y_temp_train.npy', Y_temp_train)
    np.save('data/X_temp_test.npy', X_temp_test)
    np.save('data/X_coordinate_test.npy', X_coordinate_test)
    np.save('data/Y_temp_test.npy', Y_temp_test)


if __name__ == '__main__':
    generate_npy()
