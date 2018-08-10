# --coding:utf-8 --
from alpha_eval import Alpha_eval
import time
import matplotlib.pyplot as plt
import os  
import sys
reload(sys) 
sys.setdefaultencoding('utf-8')

time0 = time.time()

# 0 表示空头、1表示多头、10表示多空、5表示5分位

workspace_path = r'''E:\ifund'''

data_path    = workspace_path+r'''\data_eval'''
alpha_path   = workspace_path+r'''\alpha_result'''
position_path= workspace_path+r'''\position'''
pnl_path     = workspace_path+r'''\PNL'''
IC_path      = workspace_path+r'''\IC'''
stocks_path  = workspace_path+r'''\stocks_choosen'''
plot_path    = workspace_path+r'''\plot'''
result_path  = workspace_path+r'''\level_retn'''

config_eval  = {

############可配置参数#####################
   'cycle':1,  
   'equal_weight': 0,  #1 等权， 2 市值平方根加权， 0 因子权重

   'res_flag'  :1,   #0 不中性，1 全中性，2 市值行业中性 ，3 行业中性
   'alpha_name': 'alpha_worldquant101_51', 
   'decay_days': 5, #EMA衰减天数

   'begin_date': '20140501',
   'end_date'  : '20170701',
   'inverse_flag'  :0,
   'save_plot_flag':1,
   'vwap_flag':1,    #1表示vwap调仓  0表示open调仓
   'eva_type'  : 10, #1表示纯多头测试，10表示多空回测 ,5表示分5档回测
   'top_ratio' : 0.4, 
   'top_stocks': 100,
   'freq' :'daily',   # 'month','daily'
   'charge' : 0.00,  #交易费用   
   'adding_notes':u'', #自定义标签,可以写中文，一般置为空字符串
   'level_retn':1, #分档次均收益，测试单调性
#################################
   'buffer':1,        
   'alpha_from_csv':1,
   'alpha_delay':0,     
   'updown_stop_flag':1,
   'suspend_flag':1,
   'trade_flag':0,
   'cixin_ST_flag':0,
   'keep_flag':0,
   'rank_flag':0,
   'position_flag':1, 
   'chgrate_regress_flag':0,
   'data_path' :data_path,
   'alpha_path':alpha_path,
   'pnl_path':pnl_path,
   'IC_path':IC_path,
   'stocks_path':stocks_path,
   'plot_path':plot_path,
   'position_path':position_path,
   'result_path':result_path,
   'univers':3500

}


print config_eval
print 'alpha_name = %s'%config_eval['alpha_name']
time1    = time.time()
print 'readytime = %s'%(time1-time0)

psx_eval =  Alpha_eval(config_eval)


psx_eval.load_data_hdf()#高频数据
#psx_eval.get_res_chg()#

time2 = time.time()
print 'load data time = %s'%(time2-time1)
psx_eval.start_eva()
if config_eval['level_retn']==1:
   psx_eval.count_free_retn(level=30)


#psx_eval.plot()
#plt.show()

time3  = time.time()
print 'count time = %s'%(time3-time2)
