# Didclab Throughput Optimization

## Setup
Install: ``pip3 install -r requirements.txt``  
Run : ``./boot.sh``
How to run everything for training:
1. First we need a config.ini so the optimizer can query InfluxDB.
   2. Create a section called ```[influx2]``` pass in `url=http://influxdb.onedatashare.org/` org=`OneDataShare`, token and timeout.
3. MONITORING_SERVICE_IP: This is only useable if the Transfer-Service is running with cockroachDB as the Repository. 
4. TRANSFER_SCHEDULER_IP: enables the env to launch another job directly through the scheduler
5. TRANSFER_SERVICE_URL: Is the publically reachable IP of the Transfer-Service, if on the same machine should be localhost. This is needed if your TS is using HSQL

## How to use it?
The environment for BO and VDA2C, required is the key to access your influxDB deployment.
This can be supplied through config.ini in root of the project

There are three HTTP API requests you can submit to this flask server:

1. Create Optimizer: Http: method= POST, past=/optimizer/create, body= 
```json
{
  "node_id": "transfer nodes id", #the node if the transfer-service is using
  "max_concurrency":  32, #number of connections to source and destination servers
  "max_pipesize": 16,  #number of reads to single write operation
  "max_chunksize": 32000000, #this in bytes
  "optimierType": "VDA2C", #This can be BO or VDA2C
  "file_count" : 50 # the number of files in the job
}
```
2. Input Optimizer
```json
{
  "concurrency": 1,
  "parallelism":  2, #number of connections to source and destination servers
  "pipelining": 16,  #number of reads to single write operation
  "chunksize": 32000000, #chunk size used in bytes
  "throughput":1 , #Achieved throughput
  "rtt" : 50 # this is collected through InfluxDb actually
}
```
3.Delete Optimizer
```json
{
  "node_id" = "nodeId" #same node id used in create optimizer
}
```
## References
- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
ikostriov provided a blueprint like repository on how to build clear RL models in a modular way.
- https://arxiv.org/abs/1712.02944 The transfer-service we used to optimize is from this project (OneDataShare).
