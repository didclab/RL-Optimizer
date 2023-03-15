
def smallest_throughput_rtt(last_row):
    #if rtt is 0 then it is a vfs node and using disk.
    use_write = False
    use_read = False
    write_thrpt = last_row['write_throughput'].iloc[-1]
    read_thrpt = last_row['read_throughput'].iloc[-1]
    source_rtt = last_row['source_rtt'].iloc[-1]
    dest_rtt = last_row['destination_rtt'].iloc[-1]
    rtt = max(source_rtt, dest_rtt)
    if write_thrpt <= 0:
        use_read = True
    elif read_thrpt <= 0:
        use_write = True
    else:
        if write_thrpt < read_thrpt:
            use_write = True
        else:
            use_read = True
    if use_write:
        thrpt = write_thrpt
    else:
        thrpt = read_thrpt

    return thrpt, rtt
