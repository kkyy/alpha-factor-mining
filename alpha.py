# -- coding: utf-8 --
import pandas as pd 
import numpy as np
import time
from datetime import datetime,timedelta
from alpha_base import Alpha_Base
import cx_Oracle as database
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model 
import h5py
from  copy import deepcopy as copy
import os
import shutil


class Alpha(Alpha_Base):

    def __init__(self,cf):
        Alpha_Base.__init__(self,cf)
   
    def load_data(self):
        
        print u'info 从文件中导入数据...'
        #获取起止时间                      
        begin_date = self.begin_date
        end_date   = self.end_date

        #self.data_path  = self.config['data_path']
        
        #获取行业信息
        self.ind_p = pd.read_csv(index_col = 0,filepath_or_buffer = self.data_path+r'\ind_p.csv')

        self.ind_index = self.ind_p.index

        self.filter_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\filter_p.csv')
        self.filter_p  = self.filter_p.set_index(self.filter_p.columns[0]).loc[begin_date:end_date]
        
        self.index     = self.filter_p.index
            #获取股票索引
        self.columns   = self.filter_p.columns

        if 'volume_price' in self.data_source:

            begin_volprice = '20100101'

            #self.filter_matrix  = pd.read_hdf(self.data_path+r'\filter_matrix.h5').loc[begin_volprice:end_date]
            self.filter_matrix  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\filter_matrix.csv')
            self.filter_matrix  = self.filter_matrix.set_index(self.filter_matrix.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.open_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\open_p.csv')
            self.open_p  = self.open_p.set_index(self.open_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)


            self.close_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\close_p.csv')
            self.close_p  = self.close_p.set_index(self.close_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.close_y  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\close_y.csv')
            self.close_y  = self.close_y.set_index(self.close_y.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.vwap_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\vwap_p.csv')
            self.vwap_p  = self.vwap_p.set_index(self.vwap_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.volume_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\volume_p.csv')
            self.volume_p  = self.volume_p.set_index(self.volume_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.chgRate_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\chgRate_p.csv')
            self.chgRate_p  = self.chgRate_p.set_index(self.chgRate_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.re_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\re_p.csv')
            self.re_p  = self.re_p.set_index(self.re_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.high_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\high_p.csv')
            self.high_p  = self.high_p.set_index(self.high_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.low_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\low_p.csv')
            self.low_p  = self.low_p.set_index(self.low_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.turnover_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\turnover_p.csv')
            self.turnover_p  = self.turnover_p.set_index(self.turnover_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.cap_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\cap_p.csv')
            self.cap_p  = self.cap_p.set_index(self.cap_p.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.zz500_p  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\zz500_p.csv')
            self.zz500_p  = self.zz500_p.set_index(self.zz500_p.columns[0]).loc[begin_volprice:end_date]

            df_trade_day  = pd.read_csv(index_col = 0,dtype = {1:np.str},filepath_or_buffer = self.data_path+r'\df_trade_day.csv')
            self.trade_day = list(df_trade_day.iloc[:,0])
          
            self.filter_updown  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\filter_updown.csv')
            self.filter_updown  = self.filter_updown.set_index(self.filter_updown.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            self.filter_suspend  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\filter_suspend.csv')
            self.filter_suspend  = self.filter_suspend.set_index(self.filter_suspend.columns[0]).loc[begin_volprice:end_date].reindex(columns = self.columns)

            yesterday = self.close_p.shift(periods=1, axis=0)
            self.returns_p = (self.close_p - yesterday)/yesterday
            
            #获取日期索引
        
        if 'style' in self.data_source:

            GROWTH_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\GROWTH_EXP.csv')
            self.factor_growth  = GROWTH_EXP.set_index(GROWTH_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
            
            LEVERAGE_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\LEVERAGE_EXP.csv')
            self.factor_leverage  = LEVERAGE_EXP.set_index(LEVERAGE_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
            
         
            LIQUIDITY_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\LIQUIDITY_EXP.csv')
            self.factor_liquidity  = LIQUIDITY_EXP.set_index(LIQUIDITY_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)

           
            MEDIUMTERMMOMENTUM_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\MEDIUMTERMMOMENTUM_EXP.csv')
            self.factor_med  = MEDIUMTERMMOMENTUM_EXP.set_index(MEDIUMTERMMOMENTUM_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
                     
            SHORTTERMMOMENTUM_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\SHORTTERMMOMENTUM_EXP.csv')
            self.factor_short  = SHORTTERMMOMENTUM_EXP.set_index(SHORTTERMMOMENTUM_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
           
            SIZE_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\SIZE_EXP.csv')
            self.factor_size  = SIZE_EXP.set_index(SIZE_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
           
            VALUE_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\VALUE_EXP.csv')
            self.factor_value  = VALUE_EXP.set_index(VALUE_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
           
            VOLATILITY_EXP  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.data_path+r'\stylefactor\VOLATILITY_EXP.csv')
            self.factor_vol  = VOLATILITY_EXP.set_index(VOLATILITY_EXP.columns[0]).loc[begin_date:end_date].reindex(columns = self.columns)
            

        #生成univers
        self.nan_flag  = copy(self.open_p).loc[begin_date:end_date]
        self.nan_flag[self.nan_flag>0]= 1.0   
        
        self.univers = (self.filter_p*self.nan_flag).reindex(columns = self.columns)
        #生成self_decay后的alpha值  
        self.result_res   = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        #根据日期截取数据
        self.stocks       = self.result_res.shape[1]
        #获取运算天数
        self.days         = self.result_res.shape[0]
        print u'info 数据导入成功'
   
    def alpha_daily_000(self):
        #测试
        #5日平均涨幅
        config ={}
        config['alpha_name'] = 'alpha_daily_000'
        config['decay_days'] = 10
        config['res_flag']   = 1

        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        for di in self.index: #self.index 即从begin到end每日的日期序列
            univers  = self.univers.loc[di]

            axis_now = self.trade_day.index(di)
            axis_5   = self.trade_day[axis_now-5+1] #获得前5个交易日的日期
           
            vect_alpha   = -self_mean_str(self.open_p,axis_5,di)*univers
            vect_alpha   = std_vect(vect_alpha)#standarization
            m    = result.loc[di] #pass by reference
            m[:] = vect_alpha #deepcopy to result, vect_alpha是pd.series也可以赋值
    
        return result,config
    

    def alpha_worldquant101_50(self):
        #(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        config ={}
        config['alpha_name'] = 'alpha_worldquant101_50'
        config['decay_days'] = 10
        config['res_flag']   = 1

        temp = pd.DataFrame(index=self.volume_p.index,columns=self.columns,data=np.nan)
        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)

        for di in self.volume_p.index[1000:]:
            axis_now_index = self.trade_day.index(di)
            axis_5   = self.trade_day[axis_now_index-5+1] #获得前5个交易日的日期

            vect_alpha = self_corr_str(self.volume_p.rank(), self.vwap_p.rank(), axis_5, di)
            vect_alpha = vect_alpha.rank()
            m    = temp.loc[di]
            m[:] = vect_alpha
        
        for di in self.index:
            univers  = self.univers.loc[di]

            axis_now_index = self.trade_day.index(di)
            axis_5   = self.trade_day[axis_now_index-5+1] #获得前5个交易日的日期

            vect_alpha = self_max_str(temp, axis_5, di)*univers
            vect_alpha = std_vect(vect_alpha)
            m    = result.loc[di]
            m[:] = vect_alpha

        return result,config

    def alpha_worldquant101_51(self):
        #((((delay(close,20)-delay(close,10))/10)-((delay(close,10)-close)/10)) < (-1*0.05)) ? 1 : ((-1*1)*(close-delay(close,1)))
        config ={}
        config['alpha_name'] = 'alpha_worldquant101_51'
        config['decay_days'] = 10
        config['res_flag']   = 1

        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        
        for di in self.index:
            univers  = self.univers.loc[di]

            axis_now_index = self.trade_day.index(di)
            axis_1_d  = self.trade_day[axis_now_index-1]
            axis_10_d   = self.trade_day[axis_now_index-10]
            axis_20_d   = self.trade_day[axis_now_index-20]

            d1 = (self.close_p.loc[axis_20_d] - self.close_p.loc[axis_10_d])/10
            d2 = (self.close_p.loc[axis_10_d] - self.close_p.loc[di])/10
            vect_alpha = d1 - d2
            vect_alpha[vect_alpha < -0.05] = 1
            vect_alpha[vect_alpha >= 0.05] = -1 * (self.close_p.loc[di] - self.close_p.loc[axis_1_d])
            vect_alpha = vect_alpha*univers
            m    = result.loc[di]
            m[:] = vect_alpha

        return result,config


    def alpha_worldquant101_52(self):
        #((((-1 * ts_min(low,5)) + delay(ts_min(low,5),5)) * rank(((sum(returns,240) - sum(returns,20))/220))) * ts_rank(volume,5))
        config ={}
        config['alpha_name'] = 'alpha_worldquant101_52'
        config['decay_days'] = 10
        config['res_flag']   = 1

        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        for di in self.index:
            univers  = self.univers.loc[di]

            axis_now = self.trade_day.index(di)
            axis_5   = self.trade_day[axis_now-5+1] #获得前5个交易日的日期
            axis_5_d   = self.trade_day[axis_now-5]

            axis_10   = self.trade_day[axis_now-10+1]
            axis_10_d  = self.trade_day[axis_now-10] 
            axis_20   = self.trade_day[axis_now-20+1]
            axis_220   = self.trade_day[axis_now-220+1]
            axis_240   = self.trade_day[axis_now-240+1]

            vect_alpha1 = -1 * self_min_str(self.low_p,axis_5,di) + self_min_str(self.low_p,axis_10_d,axis_5_d)
            vect_alpha2 = self_sum_str(self.returns_p,axis_240,di) - self_sum_str(self.returns_p,axis_20,di)/220
            vect_alpha = vect_alpha1 * vect_alpha2.rank() * self_tsrank_str(self.volume_p,axis_5,di)*univers
            
            m    = result.loc[di]
            m[:] = vect_alpha
    
        return result,config
    
    def alpha_worldquant101_53(self):
         #(-1 * delta((((close - low)-(high - close))/(close - low)),9))
        config ={}
        config['alpha_name'] = 'alpha_worldquant101_53'
        config['decay_days'] = 10
        config['res_flag']   = 1

        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)

    
        for di in self.index:
            univers  = self.univers.loc[di]

            axis_now = self.trade_day.index(di)
            axis_9   = self.trade_day[axis_now-9] #获得前9个交易日之前的日期
            
            now = ((self.close_p.loc[di]-self.low_p.loc[di])-(self.high_p.loc[di]-self.close_p.loc[di]))/(self.close_p.loc[di]-self.low_p.loc[di])
            now_9=((self.close_p.loc[axis_9]-self.low_p.loc[axis_9])-(self.high_p.loc[axis_9]-self.close_p.loc[axis_9]))/(self.close_p.loc[axis_9]-self.low_p.loc[axis_9])
            vect_alpha = -(now - now_9)*univers
            vect_alpha = std_vect(vect_alpha)
            m    = result.loc[di]
            m[:] = vect_alpha
        
        return result,config

    def alpha_worldquant101_54(self):
        #((-1 * ((low - clos) * (open^5))) / ((low - high) * (close^5)))
        config ={}
        config['alpha_name'] = 'alpha_worldquant101_54'
        config['decay_days'] = 10
        config['res_flag']   = 1

        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        for di in self.index:
            univers  = self.univers.loc[di]
            
            vect_alpha = -((self.close_p.loc[di]-self.low_p.loc[di])*(self.open_p.loc[di]**5))
            vect_alpha = vect_alpha/((self.low_p.loc[di]-self.high_p.loc[di])*(self.close_p.loc[di]**5))*univers
            vect_alpha = std_vect(vect_alpha)
            m    = result.loc[di]
            m[:] = vect_alpha
    
        return result,config

    def alpha_worldquant101_55(self):
        #(-1 * correlation(rank(((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12)))),rank(volume),6))
        config = {}
        config['alpha_name'] = 'alpha_worldquant101_55'
        config['decay_days'] = 10
        config['res_flag']   = 1

        temp = pd.DataFrame(index=self.close_p.index,columns=self.columns,data=np.nan)
        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        for di in self.close_p.index[1000:]:
            axis_now = self.trade_day.index(di)
            axis_12   = self.trade_day[axis_now-12+1] #获得前9个交易日的日期
            
            vect_alpha = self.close_p.loc[di] - self_min_str(self.low_p, axis_12, di)
            vect_alpha = vect_alpha/(self_max_str(self.low_p, axis_12, di) - self_min_str(self.low_p, axis_12, di))
            vect_alpha = vect_alpha.rank()
            m    = temp.loc[di]
            m[:] = vect_alpha

        for di in self.index:
            univers  = self.univers.loc[di]

            axis_now = self.trade_day.index(di)
            axis_6   = self.trade_day[axis_now-6+1] #获得前9个交易日的日期

            vect_alpha = -1 * self_corr_str(temp,self.volume_p.rank(),axis_6,di)*univers
            m    = result.loc[di]
            m[:] = vect_alpha
        return result,config
 
    def alpha_worldquant101_56(self):
        #(0 - (1*(rank((sum(returns,10)/sum(sum(returns, 2), 3)))*rank((returns * cap)))))
        config = {}
        config['alpha_name'] = 'alpha_worldquant101_56'
        config['decay_days'] = 10
        config['res_flag']   = 1

        temp = pd.DataFrame(index=self.returns_p.index,columns=self.columns,data=np.nan)
        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)

        temp = self.returns_p.rolling(window=2,min_periods=1).sum()

        for di in self.index:
            univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10   = self.trade_day[axis_now-10+1]
            axis_3   = self.trade_day[axis_now-3+1]
            
            vect_alpha1 = self_sum_str(self.returns_p,axis_10,di)/self_sum_str(temp,axis_3,di)
            vect_alpha2 = self.returns_p.loc[di] * self.cap_p.loc[di]
            vect_alpha = -1 * vect_alpha1.rank() * vect_alpha2.rank() * univers
            m    = result.loc[di]
            m[:] = vect_alpha

        return result,config

    def alpha_worldquant101_57(self):
        #-1*((close - vwap)/decay_linear(rank(ts_argmax(close,30)),2))
        config = {}
        config['alpha_name'] = 'alpha_worldquant101_57'
        config['decay_days'] = 10
        config['res_flag']   = 1

        temp = pd.DataFrame(index=self.close_p.index,columns=self.columns,data=np.nan)
        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)

        for di in self.close_p.index[1000:]:
            axis_now = self.trade_day.index(di)
            axis_30   = self.trade_day[axis_now-30+1]

            vect_alpha = self.close_p.loc[axis_30:di].idxmax().rank()
            m    = temp.loc[di]
            m[:] = vect_alpha

        for di in self.index:
            univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_2   = self.trade_day[axis_now-2+1]
            
            vect_alpha = self_decay_linear(temp,axis_2,di)
            vect_alpha = -1 * (self.close_p.loc[di] - self.vwap_p.loc[di])/vect_alpha * univers
            m    = result.loc[di]
            m[:] = vect_alpha

        return result,config

    def alpha_worldquant101_60(self):
        #-1*((2*scale(rank(((((close - low)-(high - close))/(high - low)) * volume)))) - scale(rank(ts_argmax(close,10))))
        config = {}
        config['alpha_name'] = 'alpha_worldquant101_60'
        config['decay_days'] = 10
        config['res_flag']   = 1

        result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)

        for di in self.index:
            univers  = self.univers.loc[di]
            axis_now = self.trade_day.index(di)
            axis_10   = self.trade_day[axis_now-10+1]
            
            v1 = (self.close_p.loc[di] - self.low_p.loc[di]) - (self.high_p.loc[di] - self.close_p.loc[di])
            v1 = v1/(self.high_p.loc[di] - self.low_p.loc[di]) * self.volume_p.loc[di]
            v1 = self_scale(v1.rank()) * 2
            v2 = self_scale(self.close_p.loc[axis_10:di].idxmax().rank())
            vect_alpha = -1 * (v1 - v2) * univers
            m    = result.loc[di]
            m[:] = vect_alpha

        return result,config

    def alpha_recount_000(self):
        #日内数据
        #数据源：fstcount_1min_001
        config ={}
        config['alpha_name'] = 'alpha_recount_000'
        config['decay_days'] = 15
        config['res_flag']   = 1
        #df_result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        intraday_1  = pd.read_csv(dtype = {0:np.str},filepath_or_buffer = self.intraday_path+r'\fstcount_1min_000.csv')
        intraday_1  = (intraday_1.set_index(intraday_1.columns[0]).loc[self.begin_date:self.end_date]).reindex(columns = self.columns) 

        return intraday_1,config                  



    def alpha_intraday_001(self):
        config ={}
        config['alpha_name'] = 'intraday_alpha_001'
        config['decay_days'] = 15
        config['res_flag']   = 0
        df_result  = pd.DataFrame(index=self.index,columns=self.columns,data=np.nan)
        for di in self.index:
            univers  = self.univers.loc[di]
            rank1 = np.zeros(self.stocks)       
            try:
                path = self.h5_path_5min+r'\%s.h5'%di
                daily_5min = h5py.File(path,'r')
                
            except:
                print r'err: file can not be find  date = %s'%di
                continue
            for i in range(self.stocks):
                try:
                    arr      = daily_5min[self.columns[i]][:].astype(float)                             
                except:
                    #print u'warning: date = %s, stocks= %s get daily data failed'%(self.index[di-delay],self.columns[i])
                    rank1[i] = np.nan
                    continue
                
                #######因子逻辑##############
                           
            # 处理日内数据的结果
            m    = df_result.loc[di]
            m[:] = rank1*univers            
            print di
                   
        return df_result,config



def fstcount_1min_000(arr,retn_name = 0):
    if retn_name == 1:
        config ={}
        config['alpha_name'] = 'fstcount_1min_000'
        config['decay_days'] = 15
        config['res_flag']   = 0
        return config
    else:
        #1min,上涨bar数量/（上涨+下跌bar数量),updown_ratio
        err_chg = (arr[:,4]-arr[:,1])/(arr[:,1]*1.0)
        err_posi= err_chg[err_chg>0]
        err_nega= err_chg[err_chg<0]
                       
        return (len(err_posi)*1.0)/(len(err_nega)+len(err_posi)+0.00001)  

"""def self_delta(df_source, d):
    ans = pd.DataFrame(index=df_source.index, columns=df_source.columns, data=np.nan)
        for i in range(d, df_source.iloc[:, 0].size):
            ans.loc[df_source.index[i]] = df_source.loc[df_source.index[i-d]] - df_source.loc[df_source.index[i]]
        return ans"""

def ind_rank(df_source,ind_ref):
    #行业内rank
    ind_source = ind_ref*df_source  
    rank_result= ind_source.rank(axis=1)
    max_value  = rank_result.max(axis=1)
    result = ((rank_result.T)/max_value).T
    return result.sum() 

def self_mean_str(df_source,last,now):
    #时间序列上均值
    mtx = df_source.loc[last:now]
    return mtx.mean(axis=0)

def self_decay_linear(df_source,last,now):
    #weighted moving average over the past d days with linearly dacaying weights d,d-1,d-2,....1(rescaled to sum up to 1)
    s = 0
    sum = 0
    last = int(last)
    now = int(now)
    d = now - last + 1
    for i in df_source.index[last:now+1]:
        s = s + d
        sum = sum + d * df_source.loc[str(i)]
    return sum/s

def self_scale(df_source,a=1):
    #rescaled x such that sum(abs(x)) = a(the default is a = 1)
    return (df_source * a) / abs(df_source).sum()

def self_sum_str(df_source,last,now):
    #时间序列上求和
    mtx = df_source.loc[last:now]
    return mtx.sum(axis=0)

def self_kurt_str(df_source,last,now):
    #时间序列上峰度
    mtx = df_source.loc[last:now]
    return mtx.kurtosis(axis=0)

def self_skew_str(df_source,last,now):
    #时间序列上偏度
    mtx = df_source.loc[last:now]
    return mtx.skew(axis=0)

def self_tsrank_str(df_source,last,now):
    #时间序列上ts_rank
    mtx = df_source.loc[last:now]
    return mtx.rank().loc[now]

def self_max_str(df_source,last,now):
    #时间序列上最大值
    mtx = df_source.loc[last:now]
    return mtx.max(axis=0)#求每一列的最大值，即每个股票在一段时间中的某个指标的最大值

def self_min_str(df_source,last,now):
    #时间序列上最小值
    mtx = df_source.loc[last:now]
    return mtx.min(axis=0)

def self_std_str(df_source,last,now):
    #时间序列上方差
    mtx = df_source.loc[last:now]
    return mtx.std(axis=0)

def self_sigmoid(vect):
    result = 1/(1+np.exp(-vect))
    return result

def self_rank(df_source):

    df_source = df_source.rank()
    min_value = df_source.min()
    max_value = df_source.max()
    if abs(max_value-min_value)<0.001:
        print 'rank error, the max_value = min_value'
        return df_source*0
    return (df_source-min_value)/(max_value-min_value)



def self_corr_str(df_source_1,df_source_2,last,now):
    #时间序列上相关性
    new_source_1 = df_source_1.loc[last:now]
    new_source_2 = df_source_2.loc[last:now]
    mult = new_source_1*new_source_2 #dataframe相乘是对应行点乘
    nan_flag = mult*0+1#nan不会变，有数的地方变成1
    new_source_1     = new_source_1*nan_flag
    new_source_2     = new_source_2*nan_flag
    cov = mult.mean(axis=0)-new_source_1.mean(axis=0)*new_source_2.mean(axis=0)
    std = new_source_1.std(axis=0)*new_source_2.std(axis=0)+0.0000001
    return cov/std

def self_corr_str_index(df_source_1,index,last,now):
    #备注，指数文件也是dataframe，虽然本身只有一列，仍需要先降维成series
    #时间序列上个股与指数相关性
    new_source_1 = df_source_1.loc[last:now]
    index = (index.loc[last:now]).iloc[:,0]
    mult  = (new_source_1.T*index).T
    cov = mult.mean(axis=0)-new_source_1.mean(axis=0)*index.mean(axis=0)
    std = new_source_1.std(axis=0)*index.std(axis=0)+0.0000001
    return cov/std

def std_vect(vect,level = 6):
    #备注，若其中有大量数值为0，则中位数和err同时为0，导致结果为0
    #向量标准化
    med = vect.median()
    err = (vect-med).abs().median()
    up_limite   = med+level*err
    down_limite = med-level*err
    vect[vect>up_limite]  = up_limite
    vect[vect<down_limite]= down_limite
    result = (vect-vect.mean())/vect.std()
    return result

#备注：x必须n*level 的矩阵，y是长为n的Series
def linear_regress(x,y):
    #delete the nan value and inf value    
    level = x.shape[1]
    flag_vect = copy(y)
    for i in range(level):
        flag_vect = flag_vect+ x.iloc[:,i]
    flag_vect[np.isinf(flag_vect)] = np.nan
    index_choosed = (flag_vect[np.isnan(flag_vect)==False]).index
    #print len(index_choosed)
    y_vect = y[index_choosed]
    x_mtx  = x.loc[index_choosed]

    linreg = LinearRegression()

    result = linreg.fit(x_mtx,y_vect)

    err    = y_vect-linreg.predict(x_mtx)
    #err    = y-linreg.predict(x)
    #get residuals
    r = pd.Series(index = y.index)
    r.loc[:] = err
    return r









            









    










            

















