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
    #file_names = [f for f in os.listdir(directory) if f.endswith('.xlsm')]
    file_names = ['Data/Singapore_Device_Priority_2022 - WW09']
    analyse_year = re.findall('\d+', file_names[0])[0]
    df = pd.read_excel(f'Data/Singapore_Device_Priority_{analyse_year} - Imputed.xlsx')
    return df

def get_cancelled_df(df):
    #remove cancelled items, for analysis later
    cancelled = df[df['STATUS'] == 'CANCELLED']
    df = df[~df['LMS #'].isin(cancelled['LMS #'].unique())]
    return df, cancelled

def analyse_jobs(df):
    #job ids that are used, drop those that are not completed by fi team
    completed_ids = list(df[df['STATUS'] == 'COMPLETED']['LMS #'].unique())
    handover_ids = list(df[df['PFA Start Count'] > 0]['LMS #'].unique())
    ids = list(set(completed_ids + handover_ids))
    
    #overall
    print(f"Total jobs in database: {df['LMS #'].nunique()}")
    complete_df = df[df['LMS #'].isin(ids)]

    print(f'Total completed jobs: {len(ids)}')
    print('--------------------------')

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

    
def cal_tat(complete_df):
    #calculate FI start to FI end for FI only

    # Group the data by the LMS column
    grouped_df = complete_df.groupby(['LMS #'])

    # Calculate the duration for each group by subtracting the minimum FI start date from the maximum FI end date

    turnaround = grouped_df['FI End'].max() - grouped_df['LMS Submission Date'].min()

    project_start = grouped_df['FI Start'].min() - grouped_df['LMS Submission Date'].min()

    result = pd.concat([turnaround, project_start], axis=1).reset_index()

    result.columns = ['LMS #', 'Turnaround','Start Duration']

    result['Start Duration'] = [x.days if x != pd.NaT else pd.NaT for x in result['Start Duration']]
    result['Turnaround'] = [x.days if x != pd.NaT else pd.NaT for x in result['Turnaround']]
    result['Queue'] = list(grouped_df['Total Delay'].max())
    result['Analysis'] = result['Turnaround'] - list(grouped_df['Total Delay'].max())
    complete_df = complete_df.merge(result, on='LMS #', how='left')
    #final_df['FI End'] = pd.to_datetime(final_df['FI End']).dt.to_period('m')

#     final_df = complete_df[['Priority #', 'LMS #','TYPE', 'Product', 'JOB DESCRIPTION', 'Queue', 'Analysis', 'Start Duration', 'LMS Submission Date', 'FI End']]
#     final_df = final_df.drop_duplicates()
    
    return complete_df

def save_df_to_excel(final_df, directory, title):
    file_names = ['Singapore_Device_Priority_2022 - WW09']
    #file_names = [f for f in os.listdir(directory) if f.endswith('.xlsm')]
    analyse_year = re.findall('\d+', file_names[0])[0]
    final_df.to_excel(f'Data/Singapore_Device_Priority_{analyse_year} - {title}.xlsx', index=False)
    
def run_calculation(directory):
    imputed_df = read_imputed_data(directory)
    df, cancelled = get_cancelled_df(imputed_df)
    save_df_to_excel(cancelled, directory, 'Cancelled')
    analyse_jobs(df)
    complete_df = cal_tat(df)
    save_df_to_excel(complete_df, directory, 'Calculated')