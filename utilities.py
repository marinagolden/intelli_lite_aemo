__author__ = "Mostafa Naemi"
__copyright__ = "Cornwall Insight Australia"


import datetime as dt
import pandas as pd


def postproc_period_to_datetime(df,date,duration = 30 , col ='Period'):
    """ this function converts the period  to Datetune in post-process stage
    when creating df from output of LP model solution"""

    df[col] = df[col].astype(int)
    if type(date) != dt.datetime:
        date = dt.datetime.strptime(str(date),'%Y-%m-%d')

    df['interval'] = (df[col] - 1)* duration
    df['Date'] = date
    df.insert(0 , 'Datetime' ,df['Date'] + pd.to_timedelta(df['interval'],unit='m'))
    df.drop(['Date','interval'],axis=1,inplace=True)
    return df

def datetime_to_period(df,duration = 30, col ='Datetime'):
    """ this function converts the Datetime to period """    
    start_date = sorted(df[col])[0].date()
    end_date = sorted(df[col])[-1].date()
    freq = str(duration) + 'min'
    hh_range = pd.date_range(start_date,end_date,freq = freq)
    
    df_ =  pd.DataFrame()
    df_[col] = hh_range
    df_['Date'] = df_[col].dt.date
    df_['Date'] = pd.to_datetime(df_['Date'] , format = '%Y-%m-%d') 
    df_['Period'] = ((df_[col] - df_.Date) /duration).dt.seconds /60 + 1 # map 00:00:00 to period 1
    df_.drop('Date',axis=1 , inplace=True)
    df = df.merge(df_ , on=col, how ='left')

    return df

def time_to_period(time , resolution =30):
    """ time is string hh:mm"""
    t = dt.datetime.strptime(time,'%H:%M')
    period = t.hour*60/resolution+t.minute/resolution +1 
    return period