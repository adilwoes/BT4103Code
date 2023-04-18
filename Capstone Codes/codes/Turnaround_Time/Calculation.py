import re
import os
import copy
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

def read_imputed_data(directory):
"""
    Reads the result from the Data Imputing step.

    Parameters:
    -----------
    directory : str
        Directory of Imputed data.

    Returns:
    --------
    df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Imputing step.
"""    
    df = pd.read_excel(f'Data/Singapore_Device_Priority - Imputed.xlsx')
    return df

def get_cancelled_df(df):
"""
    Gets the cancelled jobs from Imputed DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Imputing step.

    Returns:
    --------
    df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Imputing step without cancelled jobs.
    
    cancelled : pandas.DataFrame
       A pandas DataFrame of cancelled jobs.       
"""     
    #remove cancelled items, for analysis later
    cancelled = df[df['STATUS'] == 'CANCELLED']
    df = df[~df['LMS #'].isin(cancelled['LMS #'].unique())]
    cancelled['FI End Week'] = [x.isocalendar()[1] if pd.notnull(x) else pd.NaT for x in cancelled['LMS Submission Date']]
    cancelled['FI End Year'] = [x.isocalendar()[0] if pd.notnull(x) else pd.NaT for x in cancelled['LMS Submission Date']]
    return df, cancelled

def analyse_jobs(df):
"""
    Prints an overall summary of jobs analysed.

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Imputing step.
      
"""     
    #job ids that are used, drop those that are not completed by fi team
    completed_ids = list(df[df['STATUS'] == 'COMPLETED']['LMS #'].unique())
    handover_ids = list(df[df['PFA Start Count'] > 0]['LMS #'].unique())
    ids = list(set(completed_ids + handover_ids))
    
    #overall
    print('')
    print(f"Total jobs in database: {df['LMS #'].nunique()}")
    complete_df = df[df['LMS #'].isin(ids)]

    print(f'Total completed jobs: {len(ids)}')
    print('---------------------------------------------')

    #fi only
    fi_only = complete_df[complete_df['FI_Only'] == 'Yes']
    fi_only_usable = fi_only.loc[(fi_only['LMS Submission Count'] > 0) & (fi_only['FI End Count'] > 0)]
    fi_only_not_usable = fi_only.loc[(fi_only['LMS Submission Count'] == 0) | (fi_only['FI End Count'] == 0)]

    print(f"Total completed jobs by FI-only: {fi_only['LMS #'].nunique()}")
    print(f"Total completed jobs by FI-only without completed date: {fi_only[fi_only['FI End Count'] == 0]['LMS #'].nunique()}")
    print(f"Total completed jobs by FI-only without start date: {fi_only[fi_only['LMS Submission Count'] == 0]['LMS #'].nunique()}")
    print(f"Total completed jobs by FI-only with both dates: {fi_only_usable['LMS #'].nunique()}")
    print('FI-Only Jobs without completed/ start dates:')
    print(fi_only_not_usable['LMS #'].unique())
    
    print('---------------------------------------------')
    #fi-pfa
    fi_pfa_only = complete_df.loc[(complete_df['FI_Only'] == 'No') & (complete_df['PFA_Only'] == 'No') ]
    fi_pfa_only_usable = fi_pfa_only.loc[(fi_pfa_only['LMS Submission Count'] > 0) & (fi_pfa_only['FI End Count'] > 0)]
    fi_pfa_only_not_usable = fi_pfa_only.loc[(fi_pfa_only['LMS Submission Count'] == 0) | ((fi_pfa_only['FI End Count'] == 0) | (fi_pfa_only['PFA Start Count'] == 0))]

    print(f"Total completed jobs by FI-PFA: {fi_pfa_only['LMS #'].nunique()}")

    fi_pfa_no_completed_dates = fi_pfa_only.loc[(fi_pfa_only['FI End Count'] == 0) & (fi_pfa_only['PFA Start Count'] == 0)]

    print(f"Total completed jobs by FI-PFA without completed date: {fi_pfa_no_completed_dates['LMS #'].nunique()}")
    print(f"Total completed jobs by FI-PFA without start date: {fi_pfa_only[fi_pfa_only['LMS Submission Count'] == 0]['LMS #'].nunique()}")
    print(f"Total completed jobs by FI-PFA with both dates: {fi_pfa_only_usable['LMS #'].nunique()}")
    print('')

def cal_delays(final_df):
"""
    Calculates the delay of every observation.

    Parameters:
    -----------
    final_df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Imputing step.
    
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame with additional 4 columns that is used to derive the delay time.
      
"""      
    final_df['LMS Submission Count'] = final_df.groupby(['LMS #'])['LMS Submission Date'].transform(lambda x: x.count())
    final_df['FI End Count'] = final_df.groupby(['LMS #'])['FI End'].transform(lambda x: x.count())
    final_df['PFA Start Count'] = final_df.groupby(['LMS #'])['PFA Start'].transform(lambda x: x.count())
    final_df['Delay'] = final_df['FI Resume'] - final_df['FI Pause'] 

    #sanity check
    neg = final_df[final_df['Delay'] < timedelta(days = 0)]
    print(f"Please check LMS #: {neg['LMS #'].unique()}")

    for name, group in final_df.groupby(['LMS #']):
        total_delay = group['Delay'].sum()
        for index, row in group.iterrows():
            final_df.loc[index,'Total Delay'] = total_delay
    return final_df

def cal_tat(complete_df):
"""
    Calculates the turnaround, queue and analysis time of a job.

    Parameters:
    -----------
    complete_df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Imputing step.
    
    Returns:
    --------
    complete_df : pandas.DataFrame
       A pandas DataFrame with additional 5 columns (Turnaround, Queue, Analysis, FI End Week, FI End Year).
"""    
    #calculate FI start to FI end for FI only

    # Group the data by the LMS column
    grouped_df = complete_df.groupby(['LMS #'])

    # Calculate the duration for each group by subtracting the minimum FI start date from the maximum FI end date
    turnaround = grouped_df['FI End'].max() - grouped_df['LMS Submission Date'].min()
    result = pd.concat([turnaround], axis=1).reset_index()
    result.columns = ['LMS #', 'Turnaround']
    result['Turnaround'] = [int(x.days) if pd.notnull(x) else pd.NaT for x in result['Turnaround']]
    result['Queue'] = list(grouped_df['Total Delay'].max())
    result['Queue'] = [int(x.days) if pd.notnull(x) else 0 for x in result['Queue']]
    result['Analysis'] = result['Turnaround'] - result['Queue']
    complete_df = complete_df.merge(result, on='LMS #', how='left')
    complete_df['FI End Week'] = [x.isocalendar()[1] if pd.notnull(x) else pd.NaT for x in complete_df['FI End']]
    complete_df['FI End Year'] = [x.isocalendar()[0] if pd.notnull(x) else pd.NaT for x in complete_df['FI End']]
    
    return complete_df

def save_df_to_excel(final_df, title):
"""
    A function to save  DataFrame into Excel.

    Parameters:
    -----------
    final_df : pandas.DataFrame
       A pandas DataFrame to be exported to excel  .
    
    title : String
       Title of file to be saved as.       
       
    Returns:
    --------
    name : String
        The name that calculated DataFrame was saved as.
"""      
    name = f'Data/Singapore_Device_Priority - {title}.xlsx'
    final_df.to_excel(name, index=False)
    return name
    
def run_calculation(directory):
"""
    Runs all the steps required in the data calculation process.

    Parameters:
    -----------
    directory : str
        Directory of Imputed data.
"""
    imputed_df = read_imputed_data(directory)
    imputed_df = cal_delays(imputed_df)
    df, cancelled = get_cancelled_df(imputed_df)
    save_df_to_excel(cancelled, directory, 'Cancelled')
    print('1. Summary of Jobs have been processed')
    analyse_jobs(df)
    
    complete_df = cal_tat(df)
    print('2. Turnaround time has been calculated')
    name = save_df_to_excel(complete_df, directory, 'Calculated')
    print(f'3. Calculated Data is outputted in Excel \n at {name}')