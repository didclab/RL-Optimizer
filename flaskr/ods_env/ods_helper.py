import requests
import os
import json

headers = {"Content-Type": "application/json"}

COMPLETED = "COMPLETED"
STARTING = "STARTING"
STARTED = "STARTED"
STOPPING = "STOPPING"
STOPPED = "STOPPED"
FAILED = "FAILED"
ABANDONED = "ABANDONED"
UNKNOWN = "UNKNOWN"
RUNNING = "RUNNING"


class TransferApplicationParams(object):
    def __init__(self, transferNodeName, cc, pp, p, chunkSize):
        self.transferNodeName = transferNodeName
        self.concurrency = cc
        self.parallelism = p
        self.pipelining = pp
        self.chunkSize = chunkSize


class ItemInfo:
    def __init__(self, id: str = "", path: str = "", size: int = -1, chunk_size: int = 10000000):
        self.id = id
        self.path = path
        self.size = size
        self.chunkSize = chunk_size

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Source:
    def __init__(self, infoList=[], type: str = "", credentialId: str = "",
                 parentInfo: ItemInfo = ItemInfo("", "", -1)):
        self.type = type
        self.credId = credentialId
        self.infoList = infoList
        self.parentInfo = parentInfo

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Destination:
    def __init__(self, type: str = "", credentialId: str = "", parentInto: ItemInfo = ItemInfo()):
        self.type = type
        self.credId = credentialId
        self.parentInfo = parentInto

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class TransferOptions:
    def __init__(self, concurrencyThreadCount: int = 1, pipeSize: int = 1, chunkSize: int = 640000,
                 parallelThreadCount: int = 1, compress: bool = False, encrypt: bool = False, optimizer: str = "",
                 overwrite: str = "", retry: int = 1, verify: bool = False):
        self.concurrencyThreadCount = concurrencyThreadCount
        self.pipeSize = pipeSize
        self.chunkSize = chunkSize
        self.parallelThreadCount = parallelThreadCount
        self.compress = compress
        self.encrypt = encrypt
        self.optimizer = optimizer
        self.overwrite = overwrite
        self.retry = retry
        self.verify = verify

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class TransferJobRequest:
    def __init__(self, ownerId, source: Source = Source(), dest: Destination = Destination(),
                 TransfOp: TransferOptions = TransferOptions()):
        self.ownerId = ownerId
        self.source = source
        self.destination = dest
        self.options = TransfOp

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


"""
Sends an actions to the transfer node we configured the influx client with
"""
monitor_ip = os.getenv("MONITORING_SERVICE_IP", default="localhost")
sched_ip = os.getenv("TRANSFER_SCHEDULER_IP", default="localhost")
transfer_service_url = os.getenv("TRANSFER_SERVICE_URL", default="http://127.0.0.1:8092")


def send_application_params_tuple(cc, p, pp, transfer_node_name, chunkSize=0):
    cc = int(cc)
    p = int(p)
    pp = int(pp)
    url = "http://{}:8061/apply/application/params".format(sched_ip)
    params = TransferApplicationParams(transferNodeName=transfer_node_name, cc=str(
        cc), p=str(p), pp=str(pp), chunkSize=chunkSize)
    json_str = json.dumps(params.__dict__)
    return requests.put(url=url, data=json_str, headers=headers)


def query_job_batch_obj(jobId):
    url = "http://{}:8084/api/v1/meta/stat".format(monitor_ip)
    params = {"jobId": jobId}
    return requests.get(url=url, params=params, headers=headers).json()


"""
queries if the jobId is done. JobId we get from influx information.
"""


def query_if_job_done(jobId):
    if hsql_enabled:
        meta_data = query_batch_job_direct(jobId)
    else:
        meta_data = query_job_batch_obj(jobId)
    status = meta_data['status']
    if status == COMPLETED or status == FAILED or status == ABANDONED:
        return True, meta_data
    else:
        return False, meta_data


def query_if_job_running(jobId):
    if hsql_enabled:
        meta_data = query_batch_job_direct(jobId)
    else:
        meta_data = query_job_batch_obj(jobId)
    status = meta_data['status']
    if status == STARTING or status == STARTED or status == RUNNING:
        return True, meta_data
    else:
        return False, meta_data


def submit_transfer_request(batch_info_json, optimizer):
    url = "http://{}:8061/receiveRequest".format(sched_ip)
    tranferRequest = transform_batch_info_json_to_transfer_request(
        batch_info_json)
    if len(optimizer) > 0:
        tranferRequest.options.optimizer = optimizer
    return requests.post(url=url, data=transform_batch_info_json_to_transfer_request(batch_info_json).toJSON(),
                         headers=headers)


def transform_batch_info_json_to_transfer_request(batch_info_json):
    jobParameters = batch_info_json['jobParameters']
    batchSteps = batch_info_json['batchSteps']
    to = TransferOptions(concurrencyThreadCount=int(jobParameters['concurrency']),
                         pipeSize=int(jobParameters['pipelining']), chunkSize=int(jobParameters['chunkSize']),
                         parallelThreadCount=int(jobParameters['parallelism']),
                         compress=bool(jobParameters['compress']), optimizer=jobParameters["optimizer"],
                         retry=jobParameters['retry'])
    if 'encrypt' in jobParameters:
        to.encrypt = bool(jobParameters['encrypt'])
    if 'verify' in jobParameters:
        to.verify = bool(jobParameters['verify'])
    if 'overwrite' in jobParameters:
        to.overwrite = jobParameters['overwrite']
    info_list = []
    for step in batchSteps:
        step_name = step['step_name']  # get the step id
        # look up the stepName pojo in job params
        entityInfo = str(jobParameters[step_name])
        comma_separated = entityInfo.split(",")
        file_id = comma_separated[0].split("id=")[1].strip()[1:]
        path = comma_separated[1].split("path=")[1].strip()[1:]
        size = comma_separated[2].split("size=")[1].strip()
        chunkSize = comma_separated[3].split("chunkSize=")[1].strip()
        chunkSize = chunkSize[:-1].strip()
        itemInfo = ItemInfo(chunk_size=int(chunkSize),
                            path=path, id=file_id, size=int(size))
        info_list.append(itemInfo)
    source_type = jobParameters['sourceCredentialType']
    source_cred_id = jobParameters['sourceCredential']
    source_base_path = jobParameters['sourceBasePath']
    source_parent_info = ItemInfo(path=source_base_path, id=source_base_path)
    source = Source(infoList=info_list, type=source_type,
                    credentialId=source_cred_id, parentInfo=source_parent_info)

    dest_cred_type = jobParameters['destCredentialType']
    dest_cred_id = jobParameters['destCredential']
    dest_parent_info = ItemInfo(
        path=jobParameters['destBasePath'], id=jobParameters['destBasePath'])
    dest = Destination(credentialId=dest_cred_id,
                       type=dest_cred_type, parentInto=dest_parent_info)
    tr = TransferJobRequest(
        ownerId=jobParameters['ownerId'], source=source, dest=dest, TransfOp=to)
    return tr


def query_batch_job_direct(jobId, ts_url=None):
    target_url = transfer_service_url
    if ts_url is not None:
        target_url = ts_url

    url = "{}/api/v1/job/execution".format(target_url)
    params = {"jobId": jobId}
    return requests.get(url=url, params=params, headers=headers).json()


def query_job_ids_direct(ts_url=None):
    target_url = transfer_service_url
    if ts_url is not None:
        target_url = ts_url

    url = "{}/api/v1/job/ids".format(target_url)
    return requests.get(url=url, headers=headers).json()


def query_if_job_done_direct(jobId, ts_url=None):
    meta_data = query_batch_job_direct(jobId, ts_url)
    status = meta_data['status']
    if status == COMPLETED or status == FAILED or status == ABANDONED:
        return True, meta_data
    else:
        return False, meta_data


def query_if_job_running_direct(jobId, ts_url=None):
    meta_data = query_batch_job_direct(jobId)
    status = meta_data['status']
    if status == STARTING or status == STARTED or status == RUNNING:
        return True, meta_data
    else:
        return False, meta_data
