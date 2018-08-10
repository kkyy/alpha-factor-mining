# -- coding: utf-8 --

from Corr_check import Corr_check
from datetime import datetime
update_day  = datetime.now().strftime('%Y%m%d %H:%M:%S')


workspace_path = r'E:\ifund'

root_path      = workspace_path+r'\corr_check'
alpha_path     = workspace_path+r'\position' #被测试因子路径
alpha_ic_path  = workspace_path+r'\corr_check\IC_lib_vwap'
lib_path       = workspace_path+r'\corr_check\factor_lib'  
#alpha_path     = lib_path
IC_lib         = workspace_path+r'\corr_check\IC_lib_vwap'
 
config ={ 
 'begin_date':'20150101',
 'end_date'  :'20180628',
 
 'alpha_name':"RA_alpha_worldquant101_51_EMA5", #填alpha 因子序列，若有换行，需前一行末加\，下一行顶格

 'IC_name':u"RA_alpha_worldquant101_50_IC_vwap", #可以输入中文

 'function_set':'check_alpha',  # check_alpha 测试因子相关性，,check_ic 测试ic相关性,res_alpha 表示剔除高相关因子,muilt_process 多线程、多因子批量测试相关性
 'root_path': root_path,
 'alpha_path': alpha_path,  
 'alpha_ic_path':alpha_ic_path,
 'lib_path':lib_path,  #因子库路径
 'IC_lib':IC_lib,
 'thread_num':3, #一般设置不超过5，CPU会占用100%
 'high_corr_list':r"RA_alpha_daily_005_EMA15",
 }



if config['function_set']=='check_alpha':
	print config['alpha_name']
	Corr_check.count_corr(config)
if config['function_set']=='muilt_process':	
	if __name__ == '__main__':
		print 'run_time : %s'%update_day
		print config['alpha_name']
		Corr_check.count_corr_muiltprocess(config)
elif config['function_set']=='res_alpha':
	print config['alpha_name']
	Corr_check.high_corr_res(config)
    
elif config['function_set']=='check_ic':
	print config['IC_name']

	Corr_check.count_corr_ic(config)

else:
	print 'error: the function_set in config must be res_alpha, check_alpha or check_ic '


#Corr_check.high_corr_res(config)
