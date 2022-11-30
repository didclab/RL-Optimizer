#!/bin/bash

display_help() {
  echo "Usage: $0 [option...] (parallel | concurrency | opt_bench) <source_path> <destination_remote_id> <number_of_rounds> (<max_concurrency>| <max_parallelism>)" >&2
  echo "   opt_bench                  Command that runs the optimizer benchmark, supports BO, and VDA2C"
  echo "   parallel                   Command that runs the parallelism benchmark"
  echo "   concurrency                Command that runs the concurrency benchmark"
  echo "   <source_path>              Positional argument that is the path on the relative or absolute path to the file or folder you wish to transfer [default: ~/testData/]"
  echo "   <destination_remote_id>    Positional argument that accepts an id used in the rclone config, ~/.ssh/config hostname and onedatashare.org credential id. The id must be the same for: rclone, rsync, ~/.ssh/config for this script to work [default: uc]"
  echo "   <number_of_rounds>         Positional argument that represents the total number of rounds to run: (rclone, rsync, sftp, scp, ods) in some order [default: 5] "
  echo "   <max_concurrency>          Positional argument that represents a number that is the total number of file transfers to run at the same time. [default: 32] "
  echo "   <max_parallelism>          Positional argument that represents a number that is the total number of file transfers to run at the same time. Do not use this for sftp, scp transfers. The benchmark for parallelism is: ods, rclone only and you cannot use sftp, scp for this. [default: 32] "
  # echo some stuff here for the -a or --add-options
  echo "Setup instructions: "
  echo "Ensure: rclone, and the odscli is installed"
  echo "    Sftp Transfers: Ensure you have the ~/.ssh/config file configured to the destination server. The name of this host must be the same as the name of the remoteId in rclone, rsync uses the ssh config file, and for ODS you must add this endpoint using the UI or the cli(ideally the cli I think)."
  echo "        Sftp does not support parallelism, please only use concurrency to with this type of transfer as it will just either transfer with parallelism set to 1. Thus a totally useless benchmark"
  echo "    S3 Transfers: Add the credentials using, rclone config, and the onedatashare cli of ui. S3 only uses rclone and and ods"
  echo "    Http Transfers: Not sure this is not yet done"
  echo "General: The output files generated should all be txt files in the directory this cli is run. Each file represents the std out and err messages from that tool used to transfer. "
  echo "    In the current state rsync only supports scp or sftp and nothing else. Thus this script when doing concurrency must use sftp, or scp, with parallelism you have more flexibility as its only rclone and ods"
  exit 1
}

conc_para_all_cmd=${1:-optBench}
sourcePath=${2:-"/"}
destPath=${3:-'testData/'}
total_rounds=${4:-5}
concurrency_max=${5:-32}
parallelism_max=${6:-32}
TACC_IP=129.114.109.132
ods_cli_path=${6:-'/home/cc/odscli/onedatashare.py'}
vfs_node_id=${7:-'elvisdav@buffalo.edu-didclab-elvis-uc'}

declare -A credIds=(["http"]=httpCCTacc) #this is a map in bash aka an associative array
optimizers=("BO" "VDA2C")

mkdir -p ${HOME}/output/
touch ~/.config/rclone/rclone.conf
printf "
[httpCCTacc]
type = http
url = http://${TACC_IP}:80
" >>${HOME}/.config/rclone/rclone.conf

# python3 ~/odscli/onedatashare.py addRemote cc http://${TACC_IP}:80 http --credentialId=httpCCTacc

echo 'Arguments passed: '
echo 'Source path: ' "$sourcePath"', destRemote: '"$destRemote"', destPath: '"$destPath"', Number of rounds to run: '"$total_rounds"', max concurrency value: '"$concurrency_max" 'max parallelism:'"$max_parallelism"
echo 'ODS cli path: '"$ods_cli_path"

#Arguments:
#This just prints all arguments passed to it.
function logger() {
  for i in "$@"; do
    echo "logger-: $i"
  done
}

#Table format for csv file
#tool|conc|parallelism|pipe|source|dest|rtt|iperf3(thpt)|startTime|endTime|jobSize(bytes)|fileCount|
function csv() {
  local items=("$@")
  (
    IFS=,
    echo "${items[*]}"
  )
}

function rclone_download() {
  local start_seconds=$SECONDS
  local concurrency=$1
  local credId=$2
  local sourcePath=$3
  local parallelism=${4:-1}
  local destPath=${5:-"/home/cc/output/concurrency"}
  echo ':'$protocol ' credId:'$credId ' sourcePath:'$sourcePath ' file: /concurrency/' ' destType: vfs destRemoteId: $vfs_node_id' ' destPath:'$destPath ' concurrency:' $concurrency

  rclone copy -vP $credId:$sourcePath $destPath --transfers=$concurrency --multi-thread-streams=$parallelism --ignore-checksum --compress-level=0

  job_size=$(du -sb '/mnt/ramdisk/dest' | awk '{print $1}')
  echo 'Total Job size: '$job_size 'bytes'
  local total_seconds=$((SECONDS - start_seconds))
  local throughput=$((total_seconds / job_size))
  local mbThroughput=$((throughput * 8 / 1000))
  echo '*****rclone concurrency='$concurrency' total time: ' $total_seconds ' *****'
  echo 'bit throughput: ' $throughput ' , mb throughput:' $mbThroughput
  #    bytes     seconds                                        p pp sourceId  destRemoteId
  csv $job_size $total_seconds $mbThroughput $concurrency $parallelism 0 $credId $destRemote >>${HOME}/output/rclone_download_results.csv

}

function ods_download() {
  local concurrency=$1
  local protocol=$2
  local credId=$3
  local sourcePath=$4
  local parallelism=${5:-1}
  local destPath=${6:-"/home/cc/output/concurrency"}
  local optimizer=${7:-""}
  echo 'protocol:'$protocol ' credId:'$credId ' sourcePath:'$sourcePath ' file: /' ' destType: vfs destRemoteId: $vfs_vfs_node_id' ' destPath:'$destPath ' concurrency:' $concurrency
  python3 "$ods_cli_path" transfer "$protocol" "$credId" "$sourcePath" -f "/" vfs $vfs_node_id "$destPath" --concurrency="$concurrency" --chunksize=73383750 --compress=false --parallel="$parallelism" --pipesize=25 --optimizer="$optimizer"
  sleep 20
  python3 $ods_cli_path monitor --experiment_file="~/output/ods_download_results.csv"
}

function optimizer_bench() {
  end=5
  cc=6
  p=6
  echo "************************running optimizer bench**************************8"
  credId="nginx"
  odsCredId="tacc-http"
  for opt in "${optimizers[@]}"; do
    for ((i = 1; i <= $end; i++)); do #runs 5 transfers with BO and then 5 transfers with VDA2C
      echo 'optimizer: ' $opt ', with run round=' $i
      ods_download $cc "http" $odsCredId "/" $p "/mnt/ramdisk/dest" $opt # cc, credId, sourcePath, parallelism, destPath, optimizer
      rm -rf /mnt/ramdisk/dest
      rclone_download $cc $credId "/" $p "/mnt/ramdisk/dest" #cc, credId, sourcePath, parallelism, destPath
      rm -rf /mnt/ramdisk/dest
    done
  done
}

function concurrency_bench() {
  end=$total_rounds
  echo "**************Running concurrency bench ***************************"
  for protocol in "${!credIds[@]}"; do
    for ((i = 1; i <= $end; i++)); do
      for opt in "${optimizers[@]}"; do
        echo 'protocol ' $protocol ' credId=' ${credIds[$protocol]}
        credId=$protocol
        rm -rf ~/output/ods/concurrency/
        ods_download 6 $protocol ${credIds[$protocol]} "/" 0 "/home/cc/output/ods/concurrency"
        rm -rf ~/output/ods/concurrency/

        rm -rf ~/output/rclone/concurrency/
        rclone_download 6 ${credIds[$protocol]} "/" 0 "/home/cc/output/rclone/concurrency"
        rm -rf ~/output/rclone/concurrency/
      done
    done
  done
}

## this needs to send the output csv files to some permanent storage. Currently I will do an s3 bucket dedicated to collecting these csv files
function ods_send_output() {
  echo 'uploading output directory to pred'
  d=$(date "+%F-%T")
  python3 ${ods_cli_path} transfer vfs "$credId" /home/cc/output -f /home/cc/output/ods_download_results.csv -f /home/cc/output/rclone_download_results.csv s3 us-east-2:::odsexperimentresults $d/chameleonBenchmark/ --concurrency=2
  sleep 20
  python3 $ods_cli_path monitor
}

#the cli display handler
while [[ "$#" -gt 0 ]]; do
  case $1 in
  -h | --help)
    display_help
    shift
    ;;
  -l | --log)
    log
    shift
    ;;
  *) echo "Unknown parameter passed: $1" ;;
  esac
  shift
done

case $conc_para_all_cmd in
optBench)
  optimizer_bench
  ods_send_output
  ;;
concurrency)
  concurrency_bench
  ods_send_output
  ;;
parallelism)
  echo -n "Parallelism benchmarking not yet implemented"
  ;;
all)
  echo -n "All is not yet implemented"
  ;;

esac
