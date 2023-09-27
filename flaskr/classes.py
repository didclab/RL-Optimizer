class CreateOptimizerRequest(object):
    def __init__(self, node_id, max_concurrency, max_parallelism, max_pipesize, max_chunk_size, optimizer_type,
                 file_count, job_id, db_type, host_url=None):
        self.node_id = node_id
        self.max_concurrency = max_concurrency  # upper limit on concurrency count
        self.max_parallelism = max_parallelism  # upper limit on parallelism count
        self.max_pipesize = max_pipesize  # the max pipe size to be used
        self.max_chunksize = max_chunk_size  # max chunk size to be used
        self.optimizerType = optimizer_type
        self.file_count = file_count
        self.job_id = job_id
        self.launch_job = False
        self.db_type = db_type
        self.host_url = host_url

    def __str__(self):
        rep = "CreateOptimizerRequest object\n"
        rep += "node_id: " + self.node_id + "\n"
        rep += "max_concurrency: " + str(self.max_concurrency) + "\n"
        rep += "max_parallelism: " + str(self.max_parallelism) + "\n"
        rep += "max_pipesize: " + str(self.max_pipesize) + "\n"
        rep += "max_chunksize: " + str(self.max_chunksize) + "\n"
        rep += "optimizerType: " + str(self.optimizerType) + "\n"
        rep += "fileCount: " + str(self.file_count) + "\n"
        rep += "job_id: " + str(self.job_id) + "\n"
        rep += "db_type: " + str(self.db_type) + "\n"
        rep += "host_url: " + str(self.host_url) + "\n"
        return rep

    def set_launch_job(self, launch_job=False):
        self.launch_job = launch_job


class InputOptimizerRequest(object):
    def __init__(self, node_id, throughput, rtt, concurrency, parallelism, pipelining, chunk_size):
        self.node_id = node_id  # name of the transfer-service that is unique, this will require
        self.throughput = throughput  # reward for RL
        self.rtt = rtt
        self.concurrency = concurrency  # the concurrency thread count the transfer-service used
        self.parallelism = parallelism  # the parallelism thread count the transfer-service used
        self.pipelining = pipelining  # the pipelining size the transfer-service used
        self.chunk_size = chunk_size  # the chunk size thread count the transfer-service used

    def __str__(self):
        rep = "InputOptimizerRequest object\n"
        rep += "node_id: " + self.node_id + "\n"
        rep += "concurrency: " + str(self.concurrency) + "\n"
        rep += "parallelism: " + str(self.parallelism) + "\n"
        rep += "pipelining: " + str(self.pipelining) + "\n"
        rep += "chunk_size: " + str(self.chunk_size) + "\n"
        rep += "throughput: " + str(self.throughput) + "\n"
        rep += "rtt: " + str(self.rtt) + "\n"
        return rep


class DeleteOptimizerRequest(object):
    def __init__(self, node_id):
        self.node_id = node_id

    def __str__(self):
        rep = "DeleteOptimizerRequest object\n"
        rep += "node_id: " + self.node_id + "\n"
        return rep
