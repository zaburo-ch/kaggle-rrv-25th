import numpy as np
import pandas as pd
import base

from geopy.distance import vincenty

air_visit_data = pd.read_csv(base.INPUT_DIR + 'air_visit_data.csv', parse_dates=[1])
air_visit_data = air_visit_data.pivot(index='air_store_id', columns='visit_date', values='visitors')
air_store_info = pd.read_csv(base.INPUT_DIR + 'air_store_info.csv')
air_store_info = air_store_info.set_index('air_store_id').loc[air_visit_data.index, :]

weather_stations = pd.read_csv(base.INPUT_DIR + 'weather_stations.csv')
weathers = {}
for name in weather_stations['id']:
    weathers[name] = pd.read_csv(base.INPUT_DIR + '1-1-16_5-31-17_Weather/' + name + '.csv')
    assert len(weathers[name]) == 517

threshold = 25
precipitation = []
for pos in air_store_info[['latitude', 'longitude']].values:
    station_pos = weather_stations[['latitude', 'longitude']].values
    dist_from_store = [vincenty(pos, sp).km for sp in station_pos]
    order = np.argsort(dist_from_store)

    pi = np.zeros(517) * np.nan
    nb_filled = 0
    farest = -1
    for si in order:
        if dist_from_store[si] > threshold:
            break

        farest = si
        for j in range(517):
            if np.isnan(pi[j]):
                pi[j] = weathers[weather_stations.iloc[si, 0]].loc[j, 'precipitation']
                if not np.isnan(pi[j]):
                    nb_filled += 1

        if nb_filled == 517:
            break

    assert farest != -1
    print(dist_from_store[farest])

    precipitation.append(pi)

precipitation = np.asarray(precipitation, dtype=np.float32)
print('percentage of nan:', np.isnan(precipitation).mean())

base.save_df(precipitation, base.WORKING_DIR + 'precipitation.h5')