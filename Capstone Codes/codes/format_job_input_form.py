import pandas as pd
import json


# to categorise if job is under fi or pfa team
def get_team(df, ANALYSTS, current_directory):
    df['Date Finished'] = df['Date Finished'].apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))

    ANALYSTS_NEW = []

    for ana in ANALYSTS:
        name = ana.lower().replace(' ', '').replace('-', '')
        ANALYSTS_NEW.append(name)
    
    is_fi = []
    for index, row in df.iterrows():
        analysts = row['Analyst(s)'].lower().replace(' ', '').replace('-', '').replace(', ', ',')
        analysts_list = analysts.split(',')
        flag = False
        for analyst in analysts_list:
            if flag == True:
                break 
            
            for name in ANALYSTS_NEW:
                if (analyst in name or analyst == name):
                    is_fi.append(1)
                    flag = True
                    break  
        if flag == False:  
            is_fi.append(0)

    df['is_fi'] = is_fi
    df.to_excel(current_directory+"/Data/job_input_form_formatted.xlsx") 
    return df


# to aggregate for technology anomaly detection visualization
def tech_anomaly_detection_viz_agg(df, current_directory):
    columns = ["Product Name", "Failure", "Technology Node"]
    df = df[df['is_fi'] == 1]
    df = df[["Product Name", "Failure", "Technology Node", "Techniques"]]
    df = df.reset_index(drop = True)

    t = json.loads(df["Techniques"][0].replace("'", '"'))
    techniques = list(map(lambda x: json.loads(x.replace("'", '"')), df["Techniques"].tolist()))
    keys = []
    for t in techniques:
        keys.extend(list(t.keys()))
    keys = list(set(keys))

    success_rate_df = df.copy()
    for k in keys:
        l = list(map(lambda x: (1 if x[k][0] == "Yes" else 0) if k in x.keys() else None, techniques))
        success_rate_df[k] = l

    success_rate_df = success_rate_df.drop("Techniques", axis = 1)
    success_rate_df = success_rate_df.groupby(["Product Name", "Failure", "Technology Node"]).apply(lambda x: x.sum()/x.count()).reset_index()

    analysis = success_rate_df[keys].to_dict()
    for analy in analysis.keys():
        analysis[analy] = list(map(lambda x: "" if str(x) == "nan" else f"{analy}: {x * 100}%, ", list(analysis[analy].values())))
    result = pd.DataFrame(analysis).apply("".join, axis = 1)

    success_rate_df = success_rate_df[columns]
    success_rate_df["Technique Anomaly Rate"] = result

    success_rate_df.to_excel(current_directory + '/Data/tech_anomaly_detection_rate.xlsx')

def run_format(current_directory, file_directory, tab_name, ANALYSTS):
    df = pd.read_excel(current_directory+file_directory, tab_name) 
    df_team_cleaned = get_team(df, ANALYSTS, current_directory)
    tech_anomaly_detection_viz_agg(df_team_cleaned, current_directory)

