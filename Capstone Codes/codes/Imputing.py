import re
import os
import copy
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

month_dict = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr':4,
     'May':5,
     'Jun':6,
     'Jul':7,
     'Aug':8,
     'Sep':9,
     'Oct':10,
     'Nov':11,
     'Dec':12
    }


def read_cleaned_data(directory):
    file_names = ['Singapore_Device_Priority_2022 - WW09']
    #file_names = [f for f in os.listdir(directory) if f.endswith('.xlsm')]
    analyse_year = re.findall('\d+', file_names[0])[0]
    
    df = pd.read_excel(f'Data/Singapore_Device_Priority_{analyse_year} - Cleaned.xlsx')
    data_update = pd.read_excel(f'Data/Singapore_Device_Priority_{analyse_year} - Missing Data.xlsx', sheet_name = 'Python Import')
 
    ww_calendar = pd.read_excel('Data/Work Week Calendar.xlsx')
   
    return df, ww_calendar, data_update, file_names

def update_missing_data(df, data_update, file_names):
    data_update['FI Interim'] = [y if ~pd.isnull(y) else x for x,y in zip(data_update['FI Interim'],data_update['Remarks'])]

    analyse_year = re.findall('\d+', file_names[0])[0]
    normal_data_update = data_update[data_update['FI End'] != 'Job cancelled']

    #data update for FI-only jobs (update info at status = completed observation) 
    fi_impute = [x for x in normal_data_update['LMS #'].unique() if x in df[df['FI_Only'] == 'Yes']['LMS #'].unique()]

    for job in fi_impute:
        index = df.index[(df['LMS #'] == job) & (df['STATUS'] == 'COMPLETED')]
        index = df.index[(df['LMS #'] == job) & (df['STATUS'] == 'COMPLETED')][0]
        df.loc[index,'FI Interim/ Resume'] = data_update[data_update['LMS #'] == job]['FI Interim'].values[-1]

        #correct fi start for all observations
        index_start = df.index[(df['LMS #'] == job)]
        for ind in index_start:
            df.loc[ind,'FI Start'] = data_update[data_update['LMS #'] == job]['FI Start'].values[-1]
            df.loc[ind,'LMS Submission Date'] = data_update[data_update['LMS #'] == job]['LMS Submission Date'].values[-1]
            df.loc[ind,'FI End'] = data_update[data_update['LMS #'] == job]['FI End'].values[-1]

    #data update for FI-PFA jobs (update info at last fi job before pfa takes over)
    fipfa_impute = [x for x in normal_data_update['LMS #'].unique() if x not in fi_impute]
    fi_pfa_grp = df[df['FI_Only'] == 'No'].groupby(['LMS #'], as_index=False)

    for name, group in fi_pfa_grp:
        if name in fipfa_impute:
            c = group.iloc[0]['all_grp'].count('FI')
            fi_end = group.iloc[c-1]
            index = fi_end.name
            df.loc[index,'FI Interim/ Resume'] = data_update[data_update['LMS #'] == name]['FI Interim'].values[-1]

            #correct fi start for all observations
            index_start = df.index[(df['LMS #'] == name)]
            for ind in index_start:
                df.loc[ind,'FI Start'] = data_update[data_update['LMS #'] == name]['FI Start'].values[-1]
                df.loc[ind,'FI End'] = data_update[data_update['LMS #'] == name]['FI End'].values[-1]
                df.loc[ind,'LMS Submission Date'] = data_update[data_update['LMS #'] == name]['LMS Submission Date'].values[-1]

    # #remove open jobs
    ignore = df[(df['LMS Submission Date'] == 'Open') | (df['LMS Submission Date'] == 'Ignore')]['LMS #'].unique()
    df = df[~df['LMS #'].isin(ignore)]

    #change status for cancelled jobs in additional data
    df_grp = df.groupby(['LMS #'], as_index=False)
    new_cancelled = data_update[data_update['FI Start'] == 'Job cancelled']['LMS #'].unique()
    for name, group in df_grp:
        if name in new_cancelled:
            c = group.iloc[0]['all_grp'].count('FI')
            fi_end = group.iloc[c-1].name
            df.loc[fi_end,'STATUS'] = 'CANCELLED'
    return df

def format_dates(df, file_names):
    date_col = ['FI Start', 'FI Interim/ Resume', 'FI End', \
            'PFA Start','PFA Submission', 'LMS Submission Date',\
            'FI Pause', 'FI Resume']
    #create a new object of clean_df
    df.reset_index(inplace=True,drop=True)
    final_df = copy.deepcopy(df)

    for index, row in df.iterrows():
        new = []
        col_error = []
        for col in date_col:
            ele = row[col]
            #readable date/datetime value
            if isinstance(ele, datetime) and not pd.isnull(ele): 
                new.append(ele)
            elif isinstance(ele, str) and not ele.isalpha():
                dates = re.split('\n', ele)
                formatted_dates = []
                #iterate through to check soundness of data
                for d in dates: 
                    indiv = re.split('/', d)
                    # 2 reasons why data is not read as datetime (1. date has error 2. date not in d/m/y format)

                    #1. data has error
                    #check if year is wrong
                    year = indiv[-1]
                    analyse_year = re.findall('\d+', file_names[0])[0]
                    error = False
                    if (len(year) != 4) | (int(year) < int(analyse_year)-1) | (int(year) > int(analyse_year)+1):
                        col_error.append((dates.index(d),col,'year'))
                        formatted_dates.append(d)
                        error = True

                    #check if day/month is wrong
                    if (int(indiv[0]) > 12) & (int(indiv[1]) > 12):
                        col_error.append((dates.index(d),col,'month'))
                        formatted_dates.append(d)
                        error = True

                    #2. data not in d/m/y format
                    if not error:
                        try:
                            temp = pd.to_datetime(datetime.strptime(d,'%m/%d/%Y'))
                        except:
                            try:
                                temp = pd.to_datetime(datetime.strptime(d,'%d/%m/%Y'))
                            except:
                                ele = ele[:10]
                                temp = pd.to_datetime(datetime.strptime(d,'%Y-%m-%d')) 
                        formatted_dates.append(temp)

                if len(formatted_dates) == 1:
                    new.append(formatted_dates[0])   
                else:
                    new.append(formatted_dates)

            else:
                new.append(pd.NaT)

        #fix day/month/ year error here
        if len(col_error) > 0:
            for err in col_error:
                col_name = err[1]

                error_val = new[date_col.index(col_name)]
                is_list = False

                #check if error_val is a list (in the case of fi interim/ resume)
                if not isinstance(error_val, str):
                    error_val = error_val[err[0]]
                    is_list=True

                indiv = re.split('/', error_val)
                lms_date = new[date_col.index('LMS Submission Date')]

                #year error
                if err[2] == 'year':
                    if isinstance(lms_date, datetime) and (lms_date != pd.NaT):
                        indiv[2] = str(new[date_col.index('LMS Submission Date')].date().year)
                    corrected = '/'.join(indiv)
                    corrected_date = datetime.strptime(corrected,'%m/%d/%Y')

                    if (corrected_date.date() < new[date_col.index('LMS Submission Date')].date()) & (int(indiv[1]) < 3):
                        indiv[2] = str(new[date_col.index('LMS Submission Date')].date().year+1)
                        corrected = '/'.join(indiv)
                        corrected_date = datetime.strptime(corrected,'%m/%d/%Y')

                # day/month error (assume it is error in month)
                if err[2] == 'month':
                    if isinstance(lms_date, datetime) and (lms_date != pd.NaT):
                        indiv[0] = str(new[date_col.index('LMS Submission Date')].date().month)
                    corrected = '/'.join(indiv)
                    corrected_date = datetime.strptime(corrected,'%m/%d/%Y')

                    if (corrected_date.date() < new[date_col.index('LMS Submission Date')].date()):
                        indiv[0] = str(new[date_col.index('LMS Submission Date')].date().month+1) #rethink logic
                        corrected = '/'.join(indiv)
                        corrected_date = datetime.strptime(corrected,'%m/%d/%Y')

                if is_list:
                    new[date_col.index(col_name)][err[0]] = corrected_date
                else: 
                    new[date_col.index(col_name)] = corrected_date

        for i in range(len(date_col)):   
            update = new[i]
            if isinstance(update, list):
                update_new = ''
                for d in update:
                    update_new = update_new + d.strftime("%Y-%m-%d") + '\n'

                final_df.loc[index,date_col[i]] = update_new[:-1]
            else:
                final_df.loc[index,date_col[i]] = update
    return final_df

def fill_resume_pause(final_df):
    #fill up FI Pause and FI Resume column from either 'FI Interim/ Resume' or 'FI Pause' or 'FI Resume'
    col_list = ['FI Interim/ Resume', 'FI Pause', 'FI Resume']
    for index, row in final_df.iterrows():
        if row['GRP'] == 'FI':
            for col in col_list:
                if (pd.notnull(row[col]) & isinstance(row[col], str)):

                    #if datetime is a list
                    if '\n' in row[col]:
                        ele = re.split('\n', row[col])
                        if (row[col].count('\n') % 2 == 1):
                            final_df.loc[index, 'FI Pause'] = ele[-2]
                            final_df.loc[index, 'FI Resume'] = ele[-1]

                        #list has 3 elements, take the last element and infer date from next observation
                        else:
                            final_df.loc[index, 'FI Pause'] = ele[-1]

                    #if datetime is string
                    else: 
                        final_df.loc[index, 'FI Pause'] = row[col]

                #if value is one datetime
                elif pd.notnull(row[col]):
                    #no need to update if col is already pause/ resume
                    if (col == 'FI Interim/ Resume'):
                        final_df.loc[index, 'FI Pause'] = row[col]

    #formatting precaution 
    final_df[["FI Pause", "FI Resume", "LMS Submission Date", "FI Start", 'FI End']] = final_df[["FI Pause", "FI Resume", "LMS Submission Date", "FI Start", 'FI End']].apply(pd.to_datetime)
    
    return final_df

def format_incoming_jobs(final_df, ww_calendar):
    #handle incoming status
    incoming_df = final_df[(final_df['LMS #'] == 'INCOMING')& (final_df['GRP'] == 'FI')]
    #remove duplicate incoming status 
    incoming_df = incoming_df.drop_duplicates(subset=incoming_df.columns.difference(['Work Week']))
    final_df = final_df[(final_df['LMS #'] != 'INCOMING')]
    count=1
    for index, row in incoming_df.iterrows():
        if (pd.isnull(row['LMS Submission Date'])):
            if (pd.isnull(row['FI Start'])):
                if (pd.isnull(row['FI End'])):
                    ww = int(row['Work Week'])
                    y = int(row['Work Year'])
                    ww_calendar = ww_calendar[ww_calendar['Year'] == y]
                    m_ww = 'Dec'
                    for i in range(1,len(ww_calendar)):
                        if ww_calendar.iloc[0,i] >= m:
                            m_ww = ww_calendar.columns[i]
                            break
                    m = month_dict[m_ww]
                    
                    start_d = datetime(int(y),int(m),1)
                else: 
                    start_d = row['FI End']
            else:
                start_d = row['FI Start']
        else: 
            start_d = row['LMS Submission Date']
        row['LMS #'] = 'INCOMING' + str(count)
        count+=1
        row['INCOMING'] = 1
        row['LMS Submission Date'] = start_d
        row['FI Start'] = pd.NaT
        row['FI Interim/ Resume'] = pd.NaT
        row['FI End'] = start_d 
        row['FI Pause'] = pd.NaT    
        row['FI Resume'] = pd.NaT 
        final_df = pd.concat([final_df,pd.DataFrame(row).T.reset_index(drop=True)]).reset_index(drop=True)
        
    return final_df

def format_cancelled_jobs(final_df, ww_calendar):
    #changing cancelled jobs
    cancelled = final_df[final_df['STATUS'] == 'CANCELLED']['LMS #'].unique()
    new_df = final_df.copy(deep=True)
    final_df = final_df[~final_df['LMS #'].isin(cancelled)]

    df_grp = new_df.groupby(['LMS #'], as_index=False)

    added = pd.DataFrame()

    for name, group in df_grp:
        if name in cancelled:
            if (pd.isnull(group['FI End'].min())):
                if (pd.isnull(group['LMS Submission Date'].min())):
                    if (pd.isnull(group['FI Start'].min())):
                        ww = int(group['Work Week'].min())
                        y = int(group['Work Year'].min())
                        ww_calendar = ww_calendar[ww_calendar['Year'] == y]
                        m_ww = 'Dec'
                        for i in range(1,len(ww_calendar.columns)):
                            if ww_calendar.iloc[0,i] >= ww:
                                m_ww = ww_calendar.columns[i]
                                break
                        m = month_dict[m_ww]
                        start_d = datetime(int(y),int(m),1)
                    else: 
                        start_d = group['FI Start'].min()
                else:
                    start_d = group['LMS Submission Date'].min()
            else: 
                start_d = group['FI End'].max()
            row_data = group.iloc[0,:]
            row_data['STATUS'] = 'CANCELLED'
            row_data['LMS Submission Date'] = start_d
            row_data['FI Start'] = pd.NaT
            row_data['FI Interim/ Resume'] = pd.NaT
            row_data['FI End'] = start_d
            row_data['FI Pause'] = pd.NaT    
            row_data['FI Resume'] = pd.NaT 

            final_df = pd.concat([final_df,pd.DataFrame(row_data).T.reset_index(drop=True)]).reset_index(drop=True)
    return final_df

def infer_fi_resume(final_df):
    #fill up missing Fi Resume values with the next Fi Start value &

    temp = copy.deepcopy(final_df)
    grp_df = temp.groupby(['LMS #'], as_index=False)

    for name, group in grp_df:
        used = pd.NaT
        on_hold = True
        for index, row in group.iterrows():
            if (row['LMS #'] != 'INCOMING') & (row['STATUS'] != 'CANCELLED'):
                fi_count = row['all_grp'].count('FI')
                curr_index = list(group.index).index(index)

                if  on_hold & (pd.notnull(row['FI Pause'])):
                    final_df.loc[index,'FI Resume'] = row['FI Pause']
                    on_hold = False
                    if (pd.notnull(row['LMS Submission Date'])):
                        final_df.loc[index,'FI Pause'] = row['LMS Submission Date']
                    else: 
                        final_df.loc[index,'FI Pause'] = row['FI Start']

                #used as end resume date in previous observation
                elif final_df.loc[index,'FI Pause'] == used:
                    final_df.loc[index,'FI Pause'] = pd.NaT

                elif ((pd.notnull(row['FI Pause'])) & (pd.isnull(row['FI Resume']))):   
                    #if not last element
                    if curr_index < fi_count-1:
                        next_index = list(group.index)[curr_index+1]

                        #check if next observation has the same pause 
                        if final_df.loc[index,'FI Pause'] == final_df.loc[next_index,'FI Pause']:
                            final_df.loc[index,'FI Pause'] = pd.NaT

                        elif final_df.loc[index,'FI Pause'] > final_df.loc[next_index,'FI Start']:
                            #second last element of fi

                            if final_df.loc[index,'FI Pause'] < final_df.loc[next_index,'FI Pause']:
                                final_df.loc[index,'FI Resume'] = final_df.loc[next_index,'FI Pause']
                                used = final_df.loc[next_index,'FI Pause']

                            elif curr_index == fi_count-2:
                                if final_df.loc[index,'FI Pause'] < final_df.loc[next_index,'FI End']:
                                    final_df.loc[index,'FI Resume'] = final_df.loc[next_index,'FI End']

                            else:
                                print('oh no, please check:')
                                print(f'{name} at index {index}')

                    #if last element and fi_only
                    elif (row["FI_Only"] == 'Yes') & (curr_index == fi_count-1):
                        if final_df.loc[index,'FI Pause'] > final_df.loc[index,'FI Start']:
                            final_df.loc[index,'FI Resume'] = final_df.loc[index,'FI End']
                        else: 
                            final_df.loc[index,'FI Resume'] = final_df.loc[index,'FI Start']

                    #last fi element in list but has pfa
                    elif (row["FI_Only"] == 'No') & (curr_index == fi_count-1):

                        next_index = list(group.index)[curr_index+1]
                        if final_df.loc[index,'FI Pause'] < final_df.loc[next_index,'FI Start']:
                            final_df.loc[index,'FI Resume'] = final_df.loc[next_index,'FI Start']
                        else:
                            final_df.loc[index,'FI Resume'] = final_df.loc[next_index,'FI End']

                #if not null then remove duplicates
                elif ((pd.notnull(row['FI Pause'])) & ((pd.notnull(row['FI Resume'])) & (curr_index < fi_count-1))):
                    next_index = list(group.index)[curr_index+1]
                    if ((final_df.loc[index,'FI Pause'] == final_df.loc[next_index,'FI Pause']) & \
                    (final_df.loc[index,'FI Resume'] == final_df.loc[next_index,'FI Resume'])):
                        final_df.loc[index,'FI Pause'] = pd.NaT
                        final_df.loc[index,'FI Resume'] = pd.NaT

                if (('HOLD' not in row['STATUS']) & ('QUEUE' not in row['STATUS'])):
                    on_hold = False
                #check if it has pfa        
                if row["FI_Only"] == 'No':
                    if ((curr_index == fi_count-1) & (pd.isnull(row['FI End']))):
                        next_index = list(group.index)[curr_index+1]

                        #for case like CS0027652SG where fi info is in pfa section[fi end update]
                        if pd.notnull(final_df.loc[next_index,'FI End']):
                            final_df.loc[index,'FI End'] = final_df.loc[next_index,'FI End']

                        else: 
                            final_df.loc[index,'FI End'] = final_df.loc[next_index,'PFA Start']

                    #for case like CS0027652SG where fi info is in pfa section [fi start update]
                    if ((curr_index == fi_count-1) & (pd.isnull(row['FI Start']))):
                        if pd.notnull(final_df.loc[next_index,'FI Start']):
                            final_df.loc[index,'FI Start'] = final_df.loc[next_index,'FI Start']
                            
    #check through fi pause and resume to ensure correct year is inputed
    for index, row in final_df.iterrows():
        if (pd.notnull(row['FI Pause'])):
            if row['FI Pause'] < row['LMS Submission Date']:
                final_df.loc[index,'FI Pause'] = row['FI Pause'] + relativedelta(years=1)
        if (pd.notnull(row['FI Resume'])):
            if row['FI Resume'] < row['LMS Submission Date']:
                final_df.loc[index,'FI Resume'] = row['FI Resume'] + relativedelta(years=1)
    return final_df

def fill_all_fi_end_submission(final_df, ww_calendar):
    #fill in all rows with fi end and lms submission
    group_final_df = final_df.groupby(['LMS #'], as_index=False)
    for name, group in group_final_df:
        start_date = group['LMS Submission Date'].min()
#         if pd.isnull(start_date):
#             ww = int(group['Work Week'].min())
#             y = int(group['Work Year'].min())
#             ww_calendar = ww_calendar[ww_calendar['Year'] == y]
#             m_ww = 'Dec'
#             for i in range(1,len(ww_calendar)):
#                 if ww_calendar.iloc[0,i] >= m:
#                     m_ww = ww_calendar.columns[i]
#                     break
#             m = month_dict[m_ww]
#             start_d = datetime(int(y),int(m),1)
        end_date = group['FI End'].max()

        for index, row in group.iterrows():
            final_df.loc[index,'LMS Submission Date'] = start_date
            final_df.loc[index,'FI End'] = end_date
    return final_df

def cal_delays(final_df):
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

def save_imputed_to_excel(final_df, file_names):
    analyse_year = re.findall('\d+', file_names[0])[0]
    name = f'Data/Singapore_Device_Priority_{analyse_year} - Imputed.xlsx'
    final_df.to_excel(name, index=False)
    return name
    
def run_impute(directory):
    df, ww_calendar, data_update, file_names = read_cleaned_data(directory)
    updated_df = update_missing_data(df, data_update, file_names)
    print('1. DataFrame has been updated with the latest data')
    formatted_df = format_dates(updated_df, file_names)
    print('2. All dates are standardised')
    #formatted_df = fill_all_fi_end_submission(formatted_df, ww_calendar)
    formatted_df = fill_resume_pause(formatted_df)
    print('3. Resume and Pause Columns are filled')
    formatted_df = format_incoming_jobs(formatted_df, ww_calendar)
    print('4. Incoming Jobs are formatted')
    formatted_df = format_cancelled_jobs(formatted_df, ww_calendar)
    print('5. Cancelled Jobs are formatted')
    formatted_df = infer_fi_resume(formatted_df)
    print('6. Smart Inference excuted for FI Pause and Resume')
    formatted_df = fill_all_fi_end_submission(formatted_df, ww_calendar)
    formatted_df = cal_delays(formatted_df)
    name = save_imputed_to_excel(formatted_df, file_names)
    print(f'7. Imputed Data is outputted in Excel at \n {directory} \n as {name}')