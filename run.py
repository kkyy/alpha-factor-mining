# -- coding: utf-8 --
from alpha import Alpha
import alpha
import pandas as pd
import numpy as np
import time
time0 = time.time()

workspace_path = r'''E:\ifund'''

data_path     = workspace_path+r'''\data_source'''
result_path   = workspace_path+r'''\alpha_result'''
intraday_path = workspace_path+r'''\generate\intraday'''

h5_path_5min  = r'''E:\ifund\out_sample\H5File_5min_OS'''
h5_path_1min  = r'''E:\ifund\out_sample\H5File_1min_OS'''


config = {'begin_date': '20140328','end_date'  : '20170801',
          'data_path':data_path,
          'result_path':result_path,
          'intraday_path':intraday_path,
          'h5_path_5min':h5_path_5min,
          'h5_path_1min':h5_path_1min,
          'data_source':('volume_price','money_flow','style')    # ('volume_price','inter_day','money_flow','financial')
        }


print u'计算alpha因子'
print config

psx_alpha =  Alpha(config)

#psx_alpha.work_data(data)

psx_alpha.load_data()

time1 = time.time()
print 'load data time = %s seconds'%(time1-time0)

psx_alpha.get_new(psx_alpha.alpha_worldquant101_60) #运行alpha_daily_000的因子逻辑

#psx_alpha.get_new(psx_alpha.alpha_recount_000)
#psx_alpha.get_intraday_1min([alpha.fstcount_1min_000]) #运行日内因子,1min
#psx_alpha.get_intraday_1min([alpha.fstcount_5min_000]) #运行日内因子计算，5min

def run_5min(begin_num,end_num):
     alpha_list = []
     for i in range(begin_num,end_num+1):
          func_name = 'alpha.fstcount_5min_%03d'%i
          alpha_list = alpha_list+[eval(func_name)]

     psx_alpha.get_intraday_5min(alpha_list)

def run_1min(begin_num,end_num):
     alpha_list = []
     for i in range(begin_num,end_num+1):
          func_name = 'alpha.fstcount_1min_%03d'%i
          alpha_list = alpha_list+[eval(func_name)]
     psx_alpha.get_intraday_1min(alpha_list)

#run_1min(0,0)
time2 = time.time()
print 'count time = %s seconds'%(time2-time1)

