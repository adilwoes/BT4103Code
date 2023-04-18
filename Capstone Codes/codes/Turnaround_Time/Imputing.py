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
    		"""
        Reads all the DataFrames used in the Imputing step.

        Parameters:
        -----------
        directory : str
            Directory of input data.

        Returns:
        --------
        df : pandas.DataFrame
           A pandas DataFrame of the result from the Data Cleaning step.
           
        ww_calendar : pandas.DataFrame
           A pandas DataFrame of the Work Week Calendar.

        file_names : List
           A list containing the all the file names of the raw data.
           
        newFormat : Boolean
           A boolean value indicating whether FI Interim/ Resume column is present.
        """
    file_names = [f for f in os.listdir(directory) if f.endswith('.xlsm')]
    df = pd.read_excel(f'Data/Singapore_Device_Priority - Cleaned.xlsx')
    data_update = pd.read_excel(f'Data/Singapore_Device_Priority - Missing Data.xlsx', sheet_name = 'Python Import')
    ww_calendar = pd.read_excel('Data/Work Week Calendar.xlsx')
    if 'FI Interim/ Resume' in df.columns:
        newFormat = False
    else:
        newFormat = True
    return df, ww_calendar, data_update, file_names, newFormat

def update_missing_data(df, data_update, file_names, newFormat):
    """
    Update the cleaned data with missing data

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the result from the Data Cleaning step.
    
    data_update : pandas.DataFrame
       A pandas DataFrame of information of missing data from the Data Cleaning step.       

    file_names : List
       A list containing the all the file names of the raw data.
       
   newFormat : Boolean
       A boolean value indicating whether FI Interim/ Resume column is present.
       
    Returns:
    --------
    df : pandas.DataFrame
       A pandas DataFrame of the updated result of Data Cleaning step.
    """    
    data_update['FI Pause'] = [y if ~pd.isnull(y) else x for x,y in zip(data_update['FI Pause'], data_update['Remarks'])]

    analyse_year = re.findall('\d+', file_names[0])[0]
    normal_data_update = data_update[data_update['FI End'] != 'Job cancelled']

    #data update for FI-only jobs (update info at status = completed observation) 
    fi_impute = [x for x in normal_data_update['LMS #'].unique() if x in df[df['FI_Only'] == 'Yes']['LMS #'].unique()]

    for job in fi_impute:
        if not newFormat:
            index = df.index[(df['LMS #'] == job) & (df['STATUS'] == 'COMPLETED')][0]
            new_interim = data_update[data_update['LMS #'] == job]['FI Interim/ Resume'].values[-1]
            if pd.notnull(new_interim) & isinstance(df.loc[index,'FI Interim/ Resume'], str):
                if (new_interim != df.loc[index,'FI Interim/ Resume']):
                    df.loc[index,'FI Interim/ Resume'] = new_interim

        index_start = df.index[(df['LMS #'] == job)]
        for ind in index_start:
            df.loc[ind,'FI Start'] = data_update[data_update['LMS #'] == job]['FI Start'].values[-1]
            df.loc[ind,'LMS Submission Date'] = data_update[data_update['LMS #'] == job]['LMS Submission Date'].values[-1]
            df.loc[ind,'FI End'] = data_update[data_update['LMS #'] == job]['FI End'].values[-1]
            df.loc[ind,'FI Pause'] = data_update[data_update['LMS #'] == job]['FI Pause'].values[-1]
            df.loc[ind,'FI Resume'] = data_update[data_update['LMS #'] == job]['FI Resume'].values[-1]

    #data update for FI-PFA jobs (update info at last fi job before pfa takes over)
    fipfa_impute = [x for x in normal_data_update['LMS #'].unique() if x not in fi_impute]
    fi_pfa_grp = df[df['FI_Only'] == 'No'].groupby(['LMS #'], as_index=False)

    for name, group in fi_pfa_grp:
        if name in fipfa_impute:
            if not newFormat:
                c = group.iloc[0]['all_grp'].count('FI')
                fi_end = group.iloc[c-1]
                index = fi_end.name
                new_interim = data_update[data_update['LMS #'] == name]['FI Interim/ Resume'].values[-1]
                if pd.notnull(new_interim) & isinstance(df.loc[index,'FI Interim/ Resume'],str):
                    if (new_interim != df.loc[index,'FI Interim/ Resume']):
                        df.loc[index,'FI Interim/ Resume'] = new_interim

            index_start = df.index[(df['LMS #'] == name)]
            for ind in index_start:
                df.loc[ind,'FI Start'] = data_update[data_update['LMS #'] == name]['FI Start'].values[-1]
                df.loc[ind,'FI End'] = data_update[data_update['LMS #'] == name]['FI End'].values[-1]
                df.loc[ind,'LMS Submission Date'] = data_update[data_update['LMS #'] == name]['LMS Submission Date'].values[-1]
                df.loc[ind,'FI Pause'] = data_update[data_update['LMS #'] == name]['FI Pause'].values[-1]
                df.loc[ind,'FI Resume'] = data_update[data_update['LMS #'] == name]['FI Resume'].values[-1]

    #remove open jobs
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

def format_dates(df, file_names, newFormat):
"""
    A function to format and check on validity of date columns

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Cleaning step.  

    file_names : List
       A list containing the all the file names of the raw data.
       
   newFormat : Boolean
       A boolean value indicating whether FI Interim/ Resume column is present.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Cleaning step with corrected dates.
""" 
    date_col = ['FI Start', 'FI End', 'PFA Start','PFA Submission', 'LMS Submission Date', 'FI Pause', 'FI Resume']
    if not newFormat:
        date_col.append('FI Interim/ Resume')
    
    df.reset_index(inplace=True,drop=True)
    final_df = copy.deepcopy(df)
    df = df.groupby(['LMS #'], as_index=False)

    for name, group in df:
        for index, row in group.iterrows():
            new = []
            col_error = []
            for col in date_col:
                ele = row[col]
                if isinstance(ele, datetime):
                    # 2 reasons why data is not read as datetime (1. date has error 2. date not in d/m/y format)
                    if pd.notnull(ele):
                        #1. data has error
                        #check if year is wrong
                        year = int(ele.year)
                        error = False
                        try:
                            analyse_year = group['LMS Submission Date'].max()
                            if isinstance(analyse_year, str):
                                analyse_year = datetime.strptime(analyse_year, '%m/%d/%Y')
                            analyse_year = analyse_year.year 
                        except:
                            try:
                                analyse_year = group['FI Start'].max()
                                if isinstance(analyse_year, str) :
                                    analyse_year = datetime.strptime(analyse_year, '%m/%d/%Y')
                                analyse_year = analyse_year.year
                            except:
                                analyse_year = row['Work Year']
                        if pd.isnull(analyse_year):
                            analyse_year = row['Work Year']
                        if (list(group.index).index(index) == 0) & (year != analyse_year):
                            col_error.append((ele,col,'year'))
                            error = True
                        elif (year < int(analyse_year)-1) | (year > int(analyse_year)+1):
                            col_error.append((ele,col,'year'))
                            error = True
                    new.append(ele)
 

                elif (isinstance(ele, str) and not ele.isalpha()):
                    dates = re.split('\n', ele)
                    formatted_dates = []
                    #iterate through to check soundness of data
                    for d in dates: 
                        indiv = re.split('/', d)
                        # 2 reasons why data is not read as datetime (1. date has error 2. date not in d/m/y format)

                        #1. data has error
                        #check if year is wrong
                        year = indiv[-1]
                        try:
                            analyse_year = group['LMS Submission Date'].max()
                            if isinstance(analyse_year, str):
                                analyse_year = datetime.strptime(analyse_year, '%m/%d/%Y')
                            analyse_year = analyse_year.year 
                        except:
                            try:
                                analyse_year = group['FI Start'].max()
                                if isinstance(analyse_year, str):
                                    analyse_year = datetime.strptime(analyse_year, '%m/%d/%Y')
                                analyse_year = analyse_year.year
                            except:
                                analyse_year = row['Work Year']
                        if pd.isnull(analyse_year):
                            analyse_year = row['Work Year']
                        error = False
                        if (list(group.index).index(index) == 0) & (int(year) != analyse_year):
                            col_error.append((dates.index(d),col,'year'))
                            formatted_dates.append(d)
                            error = True
                        elif (len(year) != 4) | (int(year) < int(analyse_year)-1) | (int(year) > int(analyse_year)+1):
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
                                    try:
                                        temp = pd.to_datetime(datetime.strptime(d,'%Y-%m-%d'))
                                    except: 
                                        temp = pd.to_datetime(datetime.strptime(d,'%Y-%d-%m'))
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
                    if isinstance(error_val, list):
                        error_val = error_val[err[0]]
                        is_list=True
                    elif isinstance(error_val, datetime) & pd.notnull(error_val):
                        error_val = error_val.strftime('%m/%d/%Y')

                    indiv = re.split('/', error_val)
                    lms_date = new[date_col.index('LMS Submission Date')]

                    #year error
                    if err[2] == 'year':
                        if list(group.index).index(index) == 0:
                            indiv[2] = str(row['Work Year'])
                        elif isinstance(lms_date, datetime) and (pd.notnull(lms_date)):
                            indiv[2] = str(new[date_col.index('LMS Submission Date')].date().year)
                        else: 
                            indiv[2] = str(row['Work Year'])
                        corrected = '/'.join(indiv)
                        corrected_date = datetime.strptime(corrected,'%m/%d/%Y')
                        sub = new[date_col.index('LMS Submission Date')]
       
                        if isinstance(sub,str):
                            sub = datetime.strptime(sub,'%m/%d/%Y')
                        
                        if (corrected_date.date() < sub.date()) & (int(indiv[1]) < 3):
                            indiv[2] = str(new[date_col.index('LMS Submission Date')].date().year+1)
                            corrected = '/'.join(indiv)
                            corrected_date = datetime.strptime(corrected,'%m/%d/%Y')

                    # day/month error (assume it is error in month)
                    if err[2] == 'month':
                        if isinstance(lms_date, datetime) and (pd.notnull(lms_date)):
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

def fill_resume_pause(final_df, newFormat):
"""
    A function to fill FI Resume and FI Pause columns depending on the data type.

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Cleaning step.  
   
   newFormat : Boolean
       A boolean value indicating whether FI Interim/ Resume column is present.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame of the updated FI Pause and FI Resume columns
"""     
    #fill up FI Pause and FI Resume column from either 'FI Interim/ Resume' or 'FI Pause' or 'FI Resume'
    col_list = ['FI Pause', 'FI Resume']
    if not newFormat:
        col_list.append('FI Interim/ Resume')
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
    final_df[["FI Pause", "FI Resume", "LMS Submission Date", "FI Start", 'FI End']] = final_df[["FI Pause", "FI Resume", "LMS Submission Date", "FI Start", 'FI End']].astype('datetime64[ns]')

    if not newFormat:
        final_df = final_df.drop(columns=['FI Interim/ Resume'])
    return final_df

def format_incoming_jobs(final_df, ww_calendar):
"""
    A function to make incoming jobs unique and populate LMS Submission Date Column 

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Cleaning step.  
   
   ww_calendar : pandas.DataFrame
       A pandas DataFrame of the Work Week Calendar.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame that is updated with formatted incoming jobs.
"""     
    #handle incoming status
    incoming_df = final_df[(final_df['LMS #'] == 'INCOMING') & (final_df['GRP'] == 'FI')]
    #remove duplicate incoming status 
    incoming_df = incoming_df.drop_duplicates(subset=incoming_df.columns.difference(['Work Week']))
    final_df = final_df[(final_df['LMS #'] != 'INCOMING')]
    count=1
    for index, row in incoming_df.iterrows():
        if (pd.isnull(row['LMS Submission Date'])):
            ww = int(row['Work Week'])
            y = int(row['Work Year'])
            ww_calendar = ww_calendar[ww_calendar['Year'] == y]
            m_ww = 'Dec'
            for i in range(1,len(ww_calendar.columns)):
                if ww_calendar.iloc[0,i] >= ww:
                    m_ww = ww_calendar.columns[i]
                    break
            m = month_dict[m_ww]

            start_d = datetime(int(y),int(m),1)
        else: 
            start_d = row['LMS Submission Date']
        row['LMS #'] = 'INCOMING' + str(count)
        count+=1
        row['INCOMING'] = 1
        row['LMS Submission Date'] = start_d
        row['FI Start'] = pd.NaT
        row['FI End'] = pd.NaT 
        row['FI Pause'] = pd.NaT    
        row['FI Resume'] = pd.NaT 
        final_df = pd.concat([final_df,pd.DataFrame(row).T.reset_index(drop=True)]).reset_index(drop=True)
        
    return final_df

def format_cancelled_jobs(final_df, ww_calendar):
"""
    A function to format cancelled jobs.

    Parameters:
    -----------
    df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Cleaning step.  
   
   ww_calendar : pandas.DataFrame
       A pandas DataFrame of the Work Week Calendar.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame that is updated with formatted cancelled jobs.
"""
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
            row_data['FI End'] = pd.NaT
            row_data['FI Pause'] = pd.NaT    
            row_data['FI Resume'] = pd.NaT 

            final_df = pd.concat([final_df,pd.DataFrame(row_data).T.reset_index(drop=True)]).reset_index(drop=True)
    return final_df

def infer_fi_resume(final_df):
"""
    A function to infer FI Resume and FI Pause values from previous and next observations.

    Parameters:
    -----------
    final_df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Imputing step.  
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame that is updated with corrected FI Resume and FI Pause column.
"""    
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
                
                #for jobs immediately put on hold, impute pause and resume with lms submission and fi start
                if (curr_index == 0) & (('HOLD' in row['STATUS']) | (('QUEUE' in row['STATUS']) | ('NAN' in row['STATUS']))):
                    final_df.loc[index,'FI Pause'] = row['LMS Submission Date']
                    min_resume_null = list(group['FI Pause'])
                    min_resume = [x for x in min_resume_null if pd.notnull(x)]   
                    if (len(min_resume) > 1):
                        if (final_df.loc[index,'FI Pause'] != min_resume[1]):
                            final_df.loc[index,'FI Resume'] = min_resume[1]
                            used = min_resume[1]
                            
                    else:                            
                        final_df.loc[index,'FI Resume'] = row['FI Start']

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
                            print(final_df.loc[next_index,'FI Pause'])
                            print(final_df.loc[index,'FI Pause'])
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
                elif ((pd.notnull(row['FI Pause'])) & (((pd.notnull(row['FI Resume'])) & \
                                                       (curr_index < len(list(group.index))-1)))):
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
"""
    A function to fill all observations with FI END and LMS Submission Dates.

    Parameters:
    -----------
    final_df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Imputing step.  

    ww_calendar : pandas.DataFrame
       A pandas DataFrame of the Work Week Calendar.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A pandas DataFrame with filled FI END and LMS Submission Dates in all observations.
"""      
    #fill in all rows with fi end and lms submission
    group_final_df = final_df.groupby(['LMS #'], as_index=False)
    for name, group in group_final_df:
        submission_date = group['LMS Submission Date'].min()
        end_date = group['FI End'].max()
        start_date = group['FI Start'].min()
        if pd.isnull(submission_date):
            ww = int(group['Work Week'].min())
            y = int(group['Work Year'].min())
            ww_calendar = ww_calendar[ww_calendar['Year'] == y]
            m_ww = 'Dec'
            for i in range(1,len(ww_calendar.columns)):
                if ww_calendar.iloc[0,i] >= ww:
                    m_ww = ww_calendar.columns[i]
                    break
            m = month_dict[m_ww]
            submission_date = datetime(int(y),int(m),1)
            
        if pd.notnull(start_date):
            #validation
            if submission_date > start_date:
                submission_date = start_date
                
        elif pd.notnull(end_date):
            if submission_date > end_date:
                submission_date = end_date
                
        for index, row in group.iterrows():
            final_df.loc[index,'LMS Submission Date'] = submission_date
            final_df.loc[index,'FI End'] = end_date
            final_df.loc[index,'FI Start'] = start_date
    return final_df

def save_imputed_to_excel(final_df):
"""
    A function to save imputed DataFrame into Excel

    Parameters:
    -----------
    final_df : pandas.DataFrame
       A pandas DataFrame of the updated result from the Data Imputing step.  
       
    Returns:
    --------
    name : String
        The name that imputed DataFrame was saved as.
"""      
    name = 'Data/Singapore_Device_Priority - Imputed.xlsx'
    final_df.to_excel(name, index=False)
    return name
    
def run_impute(directory):
"""
    Runs all the steps required in the data imputation process.

    Parameters:
    -----------
    directory : str
        Directory of cleaned data.
"""
    df, ww_calendar, data_update, file_names, newFormat = read_cleaned_data(directory)
    updated_df = update_missing_data(df, data_update, file_names, newFormat)
    print('1. DataFrame has been updated with the latest data')
    formatted_df = format_dates(updated_df, file_names, newFormat)
    print('2. All dates are standardised')
    formatted_df = fill_resume_pause(formatted_df, newFormat)
    formatted_df = fill_all_fi_end_submission(formatted_df, ww_calendar)
    print('3. Resume and Pause Columns are filled')
    formatted_df = format_incoming_jobs(formatted_df, ww_calendar)
    print('4. Incoming Jobs are formatted')
    formatted_df = format_cancelled_jobs(formatted_df, ww_calendar)
    print('5. Cancelled Jobs are formatted')
    formatted_df = infer_fi_resume(formatted_df)
    print('6. Smart Inference excuted for FI Pause and Resume')
    name = save_imputed_to_excel(formatted_df)
    print(f'7. Imputed Data is outputted in Excel at \n {name}')