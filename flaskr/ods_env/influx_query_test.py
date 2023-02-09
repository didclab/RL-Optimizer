from influx_query import InfluxData

def query_db():
    influx_client = InfluxData(bucket_name="jgoldverg@gmail.com", transfer_node_name="jgoldverg@gmail.com-mac",time_window="-7d")
    df = influx_client.query_space("-7d")
    print(df)
    print(df.columns.values)


if __name__ == "__main__":
    query_db()