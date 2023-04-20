import re
import os
import copy
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

def read_data_fromDirs(directory):
    """
    Reads all Excel file for analysis.

    Parameters:
    -----------
    directory : String
       Directory of raw data.

    Returns:
    --------
    merged_df : pandas.DataFrame
       A pandas DataFrame of the merged result of all Excel for analysis.

    file_names : List
       A list of strings of file names of all Excel for analysis.
    """ 
    file_names = [f for f in os.listdir(directory) if f.endswith('.xlsm')]

    # list to store the DataFrames for each sheet
    df_list = []
    # loop through each file and read the sheet
    for file in file_names:
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, sheet_name='Device Priority')
        df['Work Week'] = int(re.split(' ',file)[-1][2:4])
        df['Work Year'] = int(re.split(' ',file)[0][-4:])
        df_list.append(df)

    # concatenate the DataFrames into a single DataFrame
    merged_df = pd.concat(df_list)
    return merged_df, file_names

def read_data_fromFiles(fileList):
    """
    Reads all Excel file for analysis from a list of file names.

    Parameters:
    -----------
    fileList : List of file names
       List comprising of file names to be analysed.
       
    Returns:
    --------
    merged_df : pandas.DataFrame
       A pandas DataFrame of the merged result of all Excel for analysis.

    file_names : List
       A list of strings of file names of all Excel for analysis.
    """ 
    # list to store the DataFrames for each sheet
    df_list = []

    # loop through each file and read the sheet
    for file in fileList:
        df = pd.read_excel(file, sheet_name='Device Priority')
        df['Work Week'] = int(re.split(' ',file)[-1][2:4])
        df['Work Year'] = int(re.split(' ',file)[0][-4:])
        df_list.append(df)
    file_names=[x.rpartition(os.sep)[-1] for x in fileList]
    # concatenate the DataFrames into a single DataFrame
    merged_df = pd.concat(df_list)
    return merged_df, file_names

def clean_data(df):
    """
    Cleans and standardise all string columns.

    Parameters:
    -----------
    df : pandas.DataFrame
       A DataFrame of all the merged Excels.
       
    Returns:
    --------
    clean_df : pandas.DataFrame
       A pandas DataFrame of the merged data frame with cleaned string columns.

    """     
    #remove empty rows
    clean_df = df[df['LMS #'].notnull()]

    #standardize values for 'Priority #'
    clean_df['Priority #'] = ['H' if x == 'H' else 'N' for x in clean_df['Priority #']]
    # Update the Priority column for all rows with the same LMS ID
    clean_df['Priority #'] = clean_df.groupby('LMS #')['Priority #'].transform('first')

    #keep only work done by pfa and fi
    clean_df = clean_df[clean_df['GRP'].isin(['PFA','FI'])]

    #capitalise and strip spaces in job id
    clean_df['LMS #'] = clean_df['LMS #'].apply(str.upper)
    clean_df['LMS #'] = clean_df['LMS #'].str.strip()

    #replace non-string values with 'NaN'
    capitalise = ['TYPE', 'Product', 'JOB DESCRIPTION', 'REQUESTOR', 'STATUS']
    for col in capitalise:
        clean_df[col] = [x if type(x) == str else 'NaN' for x in clean_df[col]]
        clean_df[col] = clean_df[col].apply(str.upper)

    #status column has different version of 'completed' eg complete 
    clean_df['STATUS'] = ['COMPLETED' if x[:8] == 'COMPLETE' else x for x in clean_df['STATUS']]
    pat = re.compile(r"\s?-\s?") # \s? matches 0 or 1 occurenece of white space

    clean_df['STATUS'] = [x if ('HOLD' not in x) & ('QUEUE' not in x) else re.sub(pat, " - ", x) for x in clean_df['STATUS']]
    clean_df['STATUS'] = [x if ('HOLD' not in x) | ('-' in x) else 'ON HOLD -' + x[7:] for x in clean_df['STATUS']]
    clean_df['STATUS'] = [x if ('QUEUE' not in x) | ('-' in x) else 'IN QUEUE -' + x[8:] for x in clean_df['STATUS']]
    
    clean_df['STATUS'] = ['CANCELLED' if x[:10] == 'CANCELLED' else x for x in clean_df['STATUS']]

    #drop irrelevant columns
    clean_df = clean_df.drop(columns=['REPORT LINK'])
    
    return clean_df

def feature_create(df):
    """
    A function to segregate if jobs are done by PFA and/ or FI Team.

    Parameters:
    -----------
    df : pandas.DataFrame
       A DataFrame of all the merged and cleaned Excels.
       
    Returns:
    --------
    df : pandas.DataFrame
       A pandas DataFrame with 3 new extra columns.

    """     
    #remove observations that has pfa only
    df['all_grp'] = df.groupby(['LMS #'])['GRP'].transform(lambda x: str(x.sum()))
    df['FI_Only'] = ['No' if 'PFA' in x else 'Yes' for x in df['all_grp']]
    df['PFA_Only'] = ['Yes' if ('PFA' in x) & ('FI' not in x) else 'No' for x in df['all_grp']]
    return df

def remove_irrelevant_rows(df):
    """
    A function to remove irrelevant PFA Only Jobs

    Parameters:
    -----------
    df : pandas.DataFrame
       A DataFrame of all the merged and cleaned Excels.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A smaller pandas DataFrame after the removal of PFA Only Jobs 

    """     
    #Remove PFA-Only Jobs
    final_df = df[df['PFA_Only'] == 'No']
    return final_df

def save_to_excel(df):
    """
    A function to saved the resultant DataFrame to Excel.

    Parameters:
    -----------
    df : pandas.DataFrame
       A DataFrame of all the merged and cleaned Excels.
       
    Returns:
    --------
    final_df : pandas.DataFrame
       A smaller pandas DataFrame after the removal of PFA Only Jobs.

    """     
    name = f'Data/Singapore_Device_Priority - Cleaned.xlsx'
    df.to_excel(name, index=False)
    return name
    
def get_jobs_missing_data(final_df):
    """
    A function to get the Job IDs of observations with missing data.

    Parameters:
    -----------
    final_df : pandas.DataFrame
       A DataFrame of all the merged and cleaned Excels.
       
    Returns:
    --------
    missing : List
       A list of Job IDs that have missing values.

    """    
    #get job ids with missing 'LMS Submission Date'
    all_ids = final_df['LMS #'].unique()
    subdate_present = final_df[pd.notnull(final_df['LMS Submission Date'])]['LMS #'].unique()
    missing_subdate =  [x for x in all_ids if x not in subdate_present]

    #get fi-only jobs with missing 'FI End Date' (only for observations with 'completed' status)
    completed_fi = final_df[(final_df['STATUS'] == 'COMPLETED') & (final_df['FI_Only'] == 'Yes')]
    enddate_present = final_df[pd.notnull(final_df['FI End'])]['LMS #'].unique()
    missing_fiend =  [x for x in completed_fi['LMS #'].unique() if x not in enddate_present] 

    #get missing end date for fi-pfa jobs
    fi_pfa = final_df[final_df['FI_Only'] == 'No']
    fi_pfa_grp = fi_pfa.groupby(['LMS #'])

    missing_end = []
    for name, group in fi_pfa_grp:
        fi_end = group['FI End'].astype('datetime64[ns]').max()
        if pd.isnull(fi_end):
            missing_end.append(group.iloc[0]['LMS #'])
    
    #save LMS # of missing data
    cancelled = list(final_df[final_df['STATUS'] == 'CANCELLED']['LMS #'].unique())
    missing = missing_subdate + missing_fiend + missing_end
    missing = [i for i in missing if i != 'INCOMING']
    missing = [i for i in missing if i not in cancelled]
    missing = set(missing)
    return missing
                          
def save_missing_to_excel(df, missing):
    """
    A function to get current available data of jobs with missing values.

    Parameters:
    -----------
    df : pandas.DataFrame
       A DataFrame of all the merged and cleaned Excels.
    
    missing : List
       A list of Job IDs that have missing values.
       
    Returns:
    --------
    name : String
       The name of Missing Data Excel file.

    """ 
    lms, start, interim, end, pause, resume, newFormat = [], [], [], [], [], [], []

    for name, group in df.groupby(['LMS #']):
        if name in missing:
            lms.append(group.iloc[-1]['LMS Submission Date'])
            start.append(group.iloc[-1]['FI Start'])
            try:
                interim.append(group.iloc[-1]['FI Interim/ Resume'])
            except:
                interim.append('')
                newFormat.append(True)
            end.append(group.iloc[-1]['FI End'])
            pause.append(group.iloc[-1]['FI Pause'])
            resume.append(group.iloc[-1]['FI Resume'])

    missing_df = pd.DataFrame(missing)
    missing_df['LMS Submission Date'] = lms
    missing_df['FI Start'] = start
    if sum(newFormat) != len(lms):
        missing_df['FI Interim/ Resume'] = interim
    missing_df['FI End'] = end
    missing_df['FI Pause'] = pause
    missing_df['FI Resume'] = resume
    missing_df['Remarks'] = [None] * len(resume)

    name = f'Data/Singapore_Device_Priority - Missing Data.xlsx'
    if sum(newFormat) != len(lms):
        missing_df.to_excel(name, index=False, header=['LMS #','LMS Submission Date','FI Start', 'FI Interim/ Resume','FI End', 'FI Pause', 'FI Resume', 'Remarks'], sheet_name='Missing Data')  
    else:
        missing_df.to_excel(name, index=False, header=['LMS #','LMS Submission Date','FI Start','FI End', 'FI Pause', 'FI Resume', 'Remarks'], sheet_name='Missing Data')  
    return name

def run_clean_fromFiles(fileNames):
    """
    A function to run all steps in the Data Cleaning Process.

    Parameters:
    -----------
    fileNames : List
       A List of filenames to be analysed

    """     
    df, file_names = read_data_fromFiles(fileNames)
    cleaned_df = clean_data(df)
    print('1. All Strings are reformatted, data is cleaned')
    final_df = feature_create(cleaned_df)
    print('2. Useful features are created')
    final_df = remove_irrelevant_rows(final_df)
    print('3. Irrelevant jobs (e.g. PFA only jobs are removed')
    name = save_to_excel(final_df, file_names)
    missing = get_jobs_missing_data(final_df)
    print(f'4. Missing Data is outputted in Excel at \n {directory} \n as {name}')
    name = save_missing_to_excel(final_df, file_names, missing, directory)  
    print(f'5. Cleaned Data is outputted in Excel at \n {directory} \n as {name}')  
    
def run_clean(directory):
    """
    A function to run all steps in the Data Cleaning Process.

    Parameters:
    -----------
    directory : String
       Directory of raw data.

    """     
    df, file_names = read_data_fromDirs(directory)
    cleaned_df = clean_data(df)
    print('1. All Strings are reformatted, data is cleaned')
    final_df = feature_create(cleaned_df)
    print('2. Useful features are created')
    final_df = remove_irrelevant_rows(final_df)
    print('3. Irrelevant jobs (e.g. PFA only jobs are removed)')
    name = save_to_excel(final_df)
    missing = get_jobs_missing_data(final_df)
    print(f'4. Missing Data is outputted in Excel \n at {name}')
    name = save_missing_to_excel(final_df, missing) 
    print(f'5. Cleaned Data is outputted in Excel \n at {name}')
