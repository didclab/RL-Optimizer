import pandas as pd
from pandas import read_csv

from influxdb_client import InfluxDBClient


class InfluxData:
    #The transfer_node_name should probably come in during the create request
    def __init__(self, bucket_name="OdsTransferNodes",transfer_node_name="jgoldverg@gmail.com-mac-mini",file_name=None, time_window="-2m"):
        self.client = InfluxDBClient.from_config_file("config.ini")
        self.space_keys = ['active_core_count, bytesDownloaded, bytesUploaded, chunkSize, concurrency, parallelism, pipelining, destination_rtt, source_rtt, read_throughput, write_throughput, ']
        self.query_api = self.client.query_api()
        self.time_window_to_query = time_window
        self.bucket_name = bucket_name
        self.transfer_node_name = transfer_node_name

        self.input_file = file_name

    def query_space(self, time_window='-2m'):
        q = '''from(bucket: "{}")
  |> range(start: {})
  |> filter(fn: (r) => r["_measurement"] == "transfer_data")
  |> filter(fn: (r) => r["APP_NAME"] == "{}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  '''.format(self.bucket_name,time_window, self.transfer_node_name)
        df = self.query_api.query_data_frame(q)
        if type(df) == list:
            df = pd.concat(df, axis=0, ignore_index=True)
        return df

    def prune_df(self, df):
        df2 = df[self.space_keys]
        return df2

    def read_file(self):
        """
        Reads CSV Influx file in working directory
        :return: Pandas dataframe
        """
        data_frame = None
        if self.input_file is not None:
            data_frame = read_csv(self.input_file)
        return data_frame

    def close_client(self):
        self.client.close()
