# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
tradingDay = '20180105'
exside_Path_ = 'D:/data/20190218/strategy/'
mainPath = './strategy/'
refPath = './ref_data/'
dataPath_ =mainPath
filelist_all = [file for file in os.listdir(dataPath_) if (file[-1] == 'v')]

#filelist = [file for file in filelist_all if (file[11] == '2')]
filelist = [file for file in filelist_all if (file[8] == '.')]
#filelist = [file for file in os.listdir('./test/') if file[-1] == 'v']\
def ReadTest(filelist):
    strategy = dict()
    wr  = pd.DataFrame()
    pnl = pd.DataFrame()
    times = pd.DataFrame()
    trade_rate = pd.DataFrame()
    
    for file in filelist:
        path = dataPath_ +file 
        #path = './test/' +file 
        strategy_df = pd.read_csv(path)
        
        strategy_df.columns = ['symbol','wr','pnl','times','todayup','trade_qty','total_qty','cumpnl']
        #strategy_df.columns = ['symbol','times','WR_excost','pnl','posRet','negRet']
    
        strategy_df.index = strategy_df.loc[:,'symbol']
        #strategy.loc[:,file[:7]] =  strategy_df.loc[:,'pnl']
        temp = pd.DataFrame(strategy_df.loc[:,'pnl'] )
        #temp = pd.DataFrame(strategy_df.loc[:,'WR_excost'] )
        temp.columns =  [file[:8]]
        pnl = pd.merge(pnl,temp,left_index = True, right_index = True, how = 'outer')
       # print(temp)
        
        temp = pd.DataFrame(strategy_df.loc[:,'wr'] )
        #temp = pd.DataFrame(strategy_df.loc[:,'WR_excost'] )
        temp.columns =  [file[:8]]
        wr = pd.merge(wr,temp,left_index = True, right_index = True, how = 'outer')
        
        temp = pd.DataFrame(strategy_df.loc[:,'times'] )
        #temp = pd.DataFrame(strategy_df.loc[:,'WR_excost'] )
        temp.columns =  [file[:8]]
        times = pd.merge(times,temp,left_index = True, right_index = True, how = 'outer')

        temp = pd.DataFrame(strategy_df.loc[:,'trade_qty'] /strategy_df.loc[:,'total_qty'])
        #temp = pd.DataFrame(strategy_df.loc[:,'WR_excost'] )
        temp.columns =  [file[:8]]
        trade_rate = pd.merge(trade_rate,temp,left_index = True, right_index = True, how = 'outer')        

        
        
    
    strategy['times'] = times
    
    strategy['wr'] = wr
    
    strategy['pnl'] = pnl
    
    pnl_cum = pnl.sum(axis =1 )
    times_cum = times.sum(axis = 1)
    wr_cum =  (wr * times ).sum(axis = 1)   / times_cum
    #wr_cum = wr.mean(axis = 1)
    
            
    (wr*times).sum(axis = 1)/times.sum(axis = 1)
    profit_day = (pnl > 0).sum(axis = 1)

    nontrade_day = (pnl == 0).sum(axis = 1)

    loss_day = (pnl < 0).sum(axis =1)

    strategy_summary = pd.DataFrame()
    strategy_summary.loc[:,'pnl'] = pnl_cum
    strategy_summary.loc[:,'times_cum'] = times_cum
    strategy_summary.loc[:,'traderate'] = trade_rate.mean(axis = 1)
    strategy_summary.loc[:,'pnl_pertrade'] = pnl_cum/times_cum
    strategy_summary.loc[:,'winrate_trade'] = wr_cum
    strategy_summary.loc[:,'winrate_day'] = profit_day/(profit_day + loss_day)
    
    strategy_summary.loc[:,'profit_%'] = profit_day/np.shape(wr)[1]
    strategy_summary.loc[:,'nontrade_%'] = nontrade_day/np.shape(wr)[1]
    strategy_summary.loc[:,'loss_%'] = loss_day/np.shape(wr)[1]
    
    strategy_summary.loc[:,'profit_day'] = profit_day
    strategy_summary.loc[:,'nontrade_day'] = nontrade_day
    strategy_summary.loc[:,'loss_day'] = loss_day
    
    return strategy_summary,pnl,times,wr,trade_rate







def multi_merge(df_list,name_,column_name):
    df = pd.DataFrame()
    for summary_ in zip(df_list,column_name):
        #print(summary_)
        df_  = summary_[0]
        name = summary_[1]
        df.loc[:,name] = df_.loc[:,name_]
    df.colomn = column_name 
    return df
    

def group_performance():
        
    '''
    随着价格变化的绘图
        
       
    计算价格分组、市值分组的绩效。
    
    '''
    high_price = data_merge.loc[:,'price']> 20
    low_price = data_merge.loc[:,'price']< 8
    h = data_merge.loc[high_price,'pnl']
    l = data_merge.loc[low_price,'pnl']
    m = data_merge.loc[(~high_price) & (~low_price),'pnl']
    print('price > 20   ,pnl:'+str(np.mean(h)))
    print('price < 20,>8,pnl:'+str(np.mean(m)))
    print('price <8     ,pnl:'+str(np.mean(l)))
    

    data_merge.loc[:,'mkt']  =np.log(data_merge.loc[:,'mkt'])
    per_75 = np.percentile( data_merge.loc[:,'mkt'] ,75)
    per_25 = np.percentile( data_merge.loc[:,'mkt'] ,25)
    high_mkt = data_merge.loc[:,'mkt']> per_75
    low_mkt = data_merge.loc[:,'mkt']< per_25
    h = data_merge.loc[high_mkt,'pnl']
    l = data_merge.loc[low_mkt,'pnl']
    m = data_merge.loc[(~high_mkt) & (~low_mkt),'pnl']
    print('mkt > 75%    ,pnl:'+str(np.mean(h)))
    print('mkt < 75,>25 ,pnl:'+str(np.mean(m)))
    print('mkt < 25     ,pnl:'+str(np.mean(l)))



       
    sort_pnl = data_merge.sort_values(by = 'price',ascending = True)
    
    plt.plot(np.log(sort_pnl.loc[:,'price']),sort_pnl.loc[:,'pnl'])
    plt.figure(num = 1)
    plt.xlabel ("log price")
    plt.ylabel ('pnl')
    
    '''
    
    
    
    
    sort_pnl = data_merge.sort_values(by = 'price',ascending = True)
    plt.plot(np.log(sort_pnl.loc[:,'price']),sort_pnl.loc[:,'times_cum'])
    plt.figure(num = 1)
    plt.xlabel ("log price")
    plt.ylabel ('times')
    '''

def month_performance():
    '''
    月度分组表现
    具体到某一日的performance
    
    '''
    
    yymmdd = list(pnl.columns)
    yymm   = list( map(lambda x: x[:6],yymmdd))
    yymmdd_df = pd.DataFrame(yymm, columns = ['yymm'])
    yymmdd_df.loc[:,'yymmdd'] = yymmdd
    yymm_dict = dict(list(yymmdd_df.groupby(['yymmdd'])))
    for key in yymm_dict.keys():
        yymm_dict[key] = list(yymm_dict[key].loc[:,'yymm'])[0]
    data = dict(list(pnl.groupby(yymm_dict,axis = 1)))
    ret = list()
    days = list()
    monthly_performance = pd.DataFrame(index = list(data.keys()))
    for key in data.keys():
        ret.append(np.mean(np.sum(data[key],axis = 1)))
        days.append(np.shape(data[key])[1])
    
    monthly_performance.loc[:,'ret']  = ret
    monthly_performance.loc[:,'days']  = days 
    np.sum(monthly_performance)
    '''
    pnl_ =  dm_o.sort_values(by = 'pnl',ascending = False)
    hp_ = pnl_.index[1:10]
    stock_profit = (pnl.loc[hp_,:])
    cum_ = stock_profit.cumsum(axis = 1)
    plt.figure()
    for i in hp_:
        plt.plot(cum_.loc[i,:])
    
    '''
    print(monthly_performance.cumsum())
    plt.plot(monthly_performance.loc[:,'ret'])
    
    
    data_times = dict(list(times.groupby(yymm_dict,axis = 1)))
    
    M1902_times = data_times['201902']

    M1902_summary = M1902_times.sum(axis = 1)
    
    
    data_wr = dict(list(wr.groupby(yymm_dict,axis = 1)))
    M1902_wr= data_wr['201902']

    M1902_summary = M1902_times.sum(axis = 1)
    
    data_pnl = dict(list(pnl.groupby(yymm_dict,axis = 1)))
    M1902_pnl= data_pnl['201902']

    M1902_summary = M1902_pnl. mean()
    print(M1902_summary)
    


if __name__ == '__main__':
    
    filelist_all = [file for file in os.listdir(dataPath_) if (file[-1] == 'v')]

    #fm_o =  [file for file in filelist_all if (file[9] == 'c')&(file[17] == 'a')]
    #fm_o =  [file for file in filelist_all if (file[9] == 't')&(file[-5] == 'z')]
    fm_o =  [file for file in filelist_all if (file[8] == 'T')]
    dm_o,pnl,times,wr,trade_rate = ReadTest(fm_o)
    '''
    ref_data = pd.read_csv(refPath + 'stock_price_mkt.csv')
    ref_data.index = ref_data.loc[:,'secucode']
    data_merge = pd.merge(dm_o,ref_data,left_index = True, right_index = True,how = 'outer')
    data_merge = data_merge.loc[pnl.index,:]
    
    
    
        
    group_performance()
    '''
    #month_performance()

    
    avg_pnl = pnl.mean()
    
    avg_pnl_cum = avg_pnl.cumsum()
    avg_pnl_cum = pnl.cumsum(axis =1 )
    
    for symbol in list(avg_pnl_cum.index):
        plt.figure()
        plt.plot(avg_pnl_cum.loc[symbol,:])
        plt.savefig( './performance/'+ 'test/' +symbol + '_1.jpg')
    
    #plt.plot(avg_pnl_cum)
    
    '''
    Rf = 1.55/ 100 / 252
    sharp_ratio = np.mean(avg_pnl - Rf) / np.std(avg_pnl)
    max_drawdown = 0 
    max_profit = 0
    for ret in avg_pnl_cum:
        max_profit = max(ret,max_profit)
        max_drawdown = max(max_drawdown,max_profit - ret)
    '''
    '''
    yymmdd = list(pnl.columns)
    yymm   = list( map(lambda x: x[:6],yymmdd))
    yymmdd_df = pd.DataFrame(yymm, columns = ['yymm'])
    yymmdd_df.loc[:,'yymmdd'] = yymmdd
    yymm_dict = dict(list(yymmdd_df.groupby(['yymmdd'])))
    for key in yymm_dict.keys():
        yymm_dict[key] = list(yymm_dict[key].loc[:,'yymm'])[0]
    data = dict(list(pnl.groupby(yymm_dict,axis = 1)))
    ret = list()
    days = list()
    monthly_performance = pd.DataFrame(index = list(data.keys()))
    for key in data.keys():
        ret.append(np.mean(np.sum(data[key],axis = 1)))
        days.append(np.shape(data[key])[1])
    
    monthly_performance.loc[:,'ret']  = ret
    monthly_performance.loc[:,'days']  = days 
    np.sum(monthly_performance)
    '''
    '''
    pnl_ =  dm_o.sort_values(by = 'pnl',ascending = False)
    hp_ = pnl_.index[1:10]
    stock_profit = (pnl.loc[hp_,:])
    cum_ = stock_profit.cumsum(axis = 1)
    plt.figure()
    for i in hp_:
        plt.plot(cum_.loc[i,:])
    
    '''
    '''
    print(monthly_performance.cumsum())
    plt.plot(monthly_performance.loc[:,'ret'])
    
    
    
    
    M1902_summary = pd.DataFrame()
    data_pnl = dict(list(pnl.groupby(yymm_dict,axis = 1)))
    M1902_pnl= data_pnl['201902']

    M1902_summary.loc[:,'pnl'] = M1902_pnl. mean()

    data_times = dict(list(times.groupby(yymm_dict,axis = 1)))
    M1902_times = data_times['201902']
    M1902_summary.loc[:,'times'] = M1902_times.sum()
    

    data_wr = dict(list(wr.groupby(yymm_dict,axis = 1)))
    M1902_wr= data_wr['201902']

    M1902_summary.loc[:,'wr'] =(M1902_wr* M1902_times).sum() / M1902_times.sum()
    
    data_tr = dict(list(trade_rate.groupby(yymm_dict,axis = 1)))
    M1902_trade_rate = data_tr['201902']
    M1902_summary.loc[:,'trade_rate'] = M1902_trade_rate.mean()   
    
    M1902_summary.loc[:,'%win_stock']= (M1902_pnl > 0).sum() / M1902_pnl.shape[0]
    M1902_summary.loc[:,'%loss_stock'] =(M1902_pnl < 0).sum() / M1902_pnl.shape[0]
    M1902_summary.loc[:,'%nontrade_stock'] =(M1902_pnl == 0).sum() / M1902_pnl.shape[0]
    
    #M1902_summary.to_csv('./'+'sh50_'+'Fer_summary.csv')
    
    
    file_Fer= [file for file in fm_o if (file[:6] == '201902')]
    dm_Fer,pnl_Fer,times_Fer,wr_Fer,trade_rate_Fer = ReadTest(file_Fer)
    '''
    ###dm_Fer.to_csv('./'+'sh50_'+'Fer_performance.csv')
    #dm_o.to_csv('./performance_sh50.csv')
    