import pandas as pd

def data_reader(filename):
    df = pd.read_csv(filename)
    time = df['utc_time']
    data = pd.pivot_table(df, index=['utc_time'], columns=['station_id_aq'],
                          values=['PM10', 'PM2.5', 'NO2', 'CO', 'O3', 'SO2'])
    data = data.values
    return data



if __name__ == "__main__":
    filename = "E:\douzeyang\intern_pingan\data\\aq_meo_gai.csv"
    a = data_reader(filename)
    print(1)