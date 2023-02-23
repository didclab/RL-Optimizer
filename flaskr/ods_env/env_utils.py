

def smallest_throughput_rtt(last_row):
    if last_row['write_throughput'].iloc[-1] < last_row['read_throughput'].iloc[-1]:
        thrpt = last_row['write_throughput'].iloc[-1]
        rtt = last_row['destination_rtt'].iloc[-1]
    else:
        thrpt = last_row['read_throughput'].iloc[-1]
        rtt = last_row['source_rtt'].iloc[-1]
    return thrpt, rtt