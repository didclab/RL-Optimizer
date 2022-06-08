# @Author: jamil
# @Date:   2021-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-07-04T14:44:57-05:00


"""
History log are grouped based on 5 fields ('FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt')
and this tuple we will denote as key, group key, group tuple
"""


import math
import random
import numpy as np
import re
import sys
import pandas as pd
from collections import Counter
from collections import deque

"""
input: a pandas dataframe and a list of fields that needs to be extracted from the csv
output: pandas dataframe from the csv with onlt the input list of required fields
"""

def extractRequiredColumn(df,requiredFields):
    return df[df.columns[df.columns.isin(requiredFields)]]

"""
input: a directory where the csv files resides could handle multiple csv files location
output: concatenated pandas datframe (in case of multiple location) or pandas dataframe
"""
def load_dataset_from_file(dataset_file_location):
    result_df=pd.read_csv(dataset_file_location[0])
    if len(dataset_file_location)>1:
        for i in range(1,len(dataset_file_location)):
            temp_df=pd.read_csv(dataset_file_location[i])
            result_df=pd.concat([result_df, temp_df], axis=0, join='inner')
    return result_df

"""
log class to store each row of the history log (only required fields) as a log
"""

class Log:
    def __init__(self, serialNo, values):
        self.serialNo = serialNo
        self.values = values
        self.ranges=[]
        self.names = ['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput','TotalEnergy','DataTransferEnergy']

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            result += "%d %s:%.2f" % (self.serialNo, self.names[i], self.values[i])
        return result

"""
Readfile Class to read a history log (.csv) and group logs based on ['FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt']
and provide group and history log as a whole statistics
"""
class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):
        self.logs=[]
        self.dataset=pd.read_csv(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        self.grouped_df=self.requiredData.groupby(['FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt'])
        self.map_for_tuple_to_throughput=dict()
        self.map_for_group_key_to_all_logs=dict()
        self.map_for_group_key_to_test_logs=dict()
        self.group_keys=[key for key, _ in self.grouped_df]
        for key,item in self.grouped_df:
            key_group_test_logs=[]
            a_group=self.grouped_df.get_group(key)
            group_max_throughput=a_group['TotalAvgTput'].max()
            self.map_for_tuple_to_throughput[key]=group_max_throughput
            number_of_rows=a_group.shape[0]
            selected_no_test_rows=math.ceil(number_of_rows*1)  #30% is test data
            a_group_test=a_group.sample(n=selected_no_test_rows)
            for index, row in a_group_test.iterrows():
                key_group_test_logs.append(Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput'],row['TotalEnergy'],row['DataTransferEnergy']]))
            self.map_for_group_key_to_test_logs[key]=key_group_test_logs

    """
    return_map_for_tuple_to_throughput() returns a dictionary for
    each group (key) of logs and max throughput for that group
    """
    def return_map_for_tuple_to_throughput(self):
        return self.map_for_tuple_to_throughput

    """
    get_grouped_df() returns a generic.DataFrameGroupBy object which needs to be accessed
    by group keys
    """
    def get_grouped_df(self):
        return self.grouped_df
    """
    return_group_specific_test_logs() returns a list of log entries associated with
    that particular group keys
    """
    def return_group_specific_test_logs(self,group_key):
        return self.map_for_group_key_to_test_logs[group_key]

    """
    return_logs() returns a list of log entries associated with
    a complete dataSet
    """
    def return_logs(self):
        for index, row in self.requiredData.iterrows():
            self.logs.append(
            Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput'],row['TotalEnergy'],row['DataTransferEnergy']]))#'P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput'
        return self.logs


"""
function takes a dataframe and max throughput and returns
max throughput corresponding CC,P and PP level as a tuple
"""

def max_throughput_to_parameter(df, max_throughput):
    for index, row in df.iterrows():
        if row['TotalAvgTput']==max_throughput:
            return (row['CC_Level'],row['P_Level'],row['PP_Level'])
        else:
            continue



"""
environmentGroup Class to build the gym environment on the history log
input: filedata_grouped_df a pandas.core.groupby.generic.DataFrameGroupBy object
       groupKey a tuple of 'FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt's
"""



class environmentGroups:
    def __init__(self,
                filedata_grouped_df,groupKey,minimum_log_number):
                self.logs=[]
                self.grouped_df=filedata_grouped_df
                self.grouping_list_name=['CC_Level','P_Level','PP_Level']
                self.a_group = pd.DataFrame()
                self.key_state_dictionary={}
                self.key_max_throughput_dictionary={}
                self.key_max_throughput_parameters={}
                self.key_group_identification={}
                self.key_group_number_of_rows={}
                self.key_group={}
                self.action_list=[]
                self.key_action_list={}
                for key in groupKey:
                    self.individual_group=self.grouped_df.get_group(key)
                    if self.individual_group.shape[0] >=minimum_log_number:
                        self.a_group = pd.concat([self.a_group,self.individual_group], axis=0)
                        ###################################
                        state_list=[]
                        action_list_individual=[]
                        group_from_grouped_df=self.individual_group.groupby(['CC_Level','P_Level','PP_Level'])#,'numActiveCores','frequency'
                        for key_2 in group_from_grouped_df.groups.keys():
                            action_list_individual.append(key_2)
                            self.action_list.append(key_2)
                            state_list.append([key[0],key[1],key[2],key[3],key[4],key_2[0],key_2[1],key_2[2]]) #,key[3],key[4]
                        self.key_state_dictionary[key]=state_list
                        self.key_max_throughput_dictionary[key]=self.individual_group['TotalAvgTput'].max()
                        self.key_max_throughput_parameters[key]=max_throughput_to_parameter(self.individual_group,self.individual_group['TotalAvgTput'].max())
                        self.key_group_identification[key]=key
                        self.key_group_number_of_rows[key]=self.individual_group.shape[0]
                        self.key_group[key]=self.individual_group
                        self.key_action_list[key]=action_list_individual
                        ####################################

                    else:
                        continue
                self.action_list_duplicate_removed = [i for n, i in enumerate(self.action_list) if i not in self.action_list[:n]]


    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the maximum throughput for the class groupkey
    """

    def group_maximum_throughput(self,key):
        return self.key_max_throughput_dictionary[key]

    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the total number of logs for the class groupkey
    """
    def total_number_of_logs(self,key):
        return self.key_group_number_of_rows[key]
    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the total dataframe for the class groupkey
    """

    def return_a_group(self,key):
        return self.key_group[key]

    """
    input:
    output:provides the group of groups (pp,p,cc) name for the class groupkey
    """

    def return_grouping_list_name(self):
        return self.grouping_list_name

    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the action list  for the class groupkey
    """
    def return_action_list(self,key):
        return self.key_action_list[key]

    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the action list  for the class groupkey
    """
    def return_global_action_list(self):
        return self.action_list_duplicate_removed

    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the state list  for the class groupkey
    """
    def return_state_list(self,key):
        return self.key_state_dictionary[key]

    """
    input: takes a tuple of action key ('CC_Level','P_Level','PP_Level')
    output:provides the list of all the throughputs for the class groupkey and
           action key ('CC_Level','P_Level','PP_Level')
    """
    def return_group_key_throughput(self,group_key,search_key):
        result_throughput=[]
        group_from_grouped_df=self.grouped_df.get_group(group_key)
#         print(group_from_grouped_df,type(group_from_grouped_df))
        group_of_group=group_from_grouped_df.groupby(['CC_Level','P_Level','PP_Level'])
        log_group=group_of_group.get_group(search_key)
#         print(log_group)
        for index, row in log_group.iterrows():
            result_throughput.append(row['TotalAvgTput'])
        return result_throughput
    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the group key as tuple ('FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt')
    """
    def return_group_identification(self,key):
        return self.key_group_identification[key]

    """
    input:groupd key with ('FileCount', 'AvgFileSize', 'BufSize', 'Bandwidth', 'AvgRtt')
    output:provides the group max throughput corresponding CC,P and PP level as a tuple
    """
    def return_group_max_throughput_parameters(self,key):
        return self.key_max_throughput_parameters[key]