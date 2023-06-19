import os

import pandas as pd
from pandas import read_csv
import time
from influxdb_client import InfluxDBClient


class InfluxData:
    # The transfer_node_name should probably come in during the create request
    def __init__(self, bucket_name="OdsTransferNodes", transfer_node_name="jgoldverg@gmail.com-mac", file_name=None, time_window="-2m"):
        self.client = InfluxDBClient(
            url=os.getenv("INFLUXDB_URL"),
            token=os.getenv("INFLUXDB_TOKEN"),
            org=os.getenv("INFLUXDB_ORG")
        )
        self.space_keys = [
            'active_core_count, bytesDownloaded, bytesUploaded, chunkSize, concurrency, parallelism, pipelining, destination_rtt, source_rtt, read_throughput, write_throughput, ']
        self.query_api = self.client.query_api()
        self.time_window_to_query = time_window
        self.bucket_name = bucket_name
        self.transfer_node_name = transfer_node_name

        self.input_file = file_name

    def query_space(self, time_window='-2m', keys_to_expect=None, retry_count=10) -> pd.DataFrame:
        if keys_to_expect is None:
            keys_to_expect = ['bytesDownloaded', 'bytesUploaded', 'chunkSize', 'concurrency', 'destination_latency',
                              'destination_rtt', 'jobSize', 'parallelism', 'pipelining', 'read_throughput',
                              'source_latency', 'source_rtt', 'write_throughput', 'jobId']
        q = '''from(bucket: "{}")
  |> range(start: {})
  |> filter(fn: (r) => r["_measurement"] == "transfer_data")
  |> filter(fn: (r) => r["APP_NAME"] == "{}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  '''.format(self.bucket_name, time_window, self.transfer_node_name)
        done = False

        df = pd.DataFrame()
        while not done:
            df = self.query_api.query_data_frame(q)
            if type(df) == list:
                df = pd.concat(df, axis=0, ignore_index=True)
            if df.empty:
                time.sleep(1)
                continue
            if not all([item in list(df.columns) for item in keys_to_expect]):
                time.sleep(1)
                continue
            else:
                done = True
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
