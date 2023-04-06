import pandas as pd
import holidays
import warnings
warnings.simplefilter("ignore")

class ProcessAutomation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.DataFrame()
        self.process()
        self.to_excel()

    def process(self):
        self.df = pd.read_excel(self.file_path, sheet_name='Hist. Data')
        self.df.columns = self.df.columns.str.lower()
        self.df['date finished'] = pd.to_datetime(self.df['date finished'])
        self.df['year_month'] = self.df['date finished'].apply(lambda x: x.strftime("%Y-%m"))

    def to_excel(self):
        self.df.to_excel(f'Data/Job_Input_Form_Cleaned.xlsx', index=False)

class TATAutomation:
	def __init__(self, file_path, file_path_cancelled, ww_fp):
		self.file_path = file_path
		self.file_path_cancelled = file_path_cancelled
		self.ww_fp = ww_fp
		self.ww = pd.DataFrame()
		self.df = pd.DataFrame()
		self.cancelled = pd.DataFrame()
		self.priority = []
		self.type = []
		self.product = []
		self.description = []
		self.workyear = []
		self.workweek = []
		self.process()
		self.process_ww()

	def process(self):
		self.df = pd.read_excel(self.file_path)
		self.cancelled = pd.read_excel(self.file_path_cancelled)
		self.priority = list(self.df['Priority #'].unique())
		self.priority.append('ALL')
		self.type = list(self.df['TYPE'].unique())
		self.type.append('ALL')
		self.description = list(self.df['JOB DESCRIPTION'].unique())
		self.description.append('ALL')
		self.product = list(self.df['Product'].unique())
		self.product.append('ALL')
		self.workyear = list(self.df['Work Year'].unique())
		self.workweek = list(self.df['Work Week'].unique())
		
	def process_ww(self):
		self.ww = pd.read_excel(self.ww_fp)
		self.ww = self.ww[self.ww['Year'].isin(self.workyear)].reset_index(drop=True)

class AnomalyAutomation:
	def __init__(self, file_path):
		self.file_path = file_path
		self.df = pd.DataFrame()
		self.product = []
		self.failure = []
		self.tech_node = []
		self.process()
		self.get_unique()

	def process(self):
		self.df = pd.read_excel(self.file_path)
		self.df = self.df.drop(self.df.columns[[0]], axis=1)
		self.df.rename(columns={'Product Name': 'product_name', 'Failure': 'failure', 'Technology Node': 'technology_node'}, inplace=True)
	
	def get_unique(self):
		self.product = self.unique_sorted_values_plus_ALL(self.df.product_name)
		self.failure = self.unique_sorted_values_plus_ALL(self.df.failure)
		self.tech_node = self.unique_sorted_values_plus_ALL(self.df.technology_node)	
	
	def unique_sorted_values_plus_ALL(self, array):
		unique = array.unique().tolist()
		unique.sort()
		unique.insert(0, 'ALL')
		return unique

class ProcessToolUtilization:

	ANALYSTS = ['david jinjie', 'hu haoran', 'nathan linarto', 
            'winson lua', 'ng kaihui', 'angeline phoa', 
            'gopinath ranganathan', 'venkat-krishnan ravikumar', 
            'seah yi-xuan', 'vasanth somasundaram', 'nicholas tee']

	NOT_UTILIZED_WORDS = ["upgrade", "pm", "repair", "test", "check"]

	def __init__(self, tool_files):
		self.tool_files = tool_files
		self.columns = ["date/timezone", "analyst", "project", "tool"]
		self.tools_df = pd.DataFrame(columns=self.columns)
		self.parse_files()
		self.processed_df = self.preprocess()
		self.final_df = self.format_util_rate_weekly()

	def is_abbreviation(self, abbr, word): 
		"""
		Function for testing if abbr is an abbreviation of word
		To be placed inside list of help functions
		"""
		if not (abbr and word):
		    return False
		i, j = 0, 0
		while i < len(abbr) and j < len(word):
		    if abbr[i].upper() == word[j].upper():
		        i += 1
		    j += 1
		if i != len(abbr) and i < len(word):
		    return False
		start_cond = abbr[0] == word[0] 
		return start_cond 

	def clean_analyst(self, analyst_name):
	    if analyst_name == "venkatram":
	        return analyst_name
	    
	    for name in self.ANALYSTS:
	        if self.is_abbreviation(analyst_name, name):
	            return name
	    return analyst_name

	def clean_project(self, project_details, word_list):
		if any(word in project_details.lower() for word in word_list):
		    return "upgrade"
		else:
			return "tool_utilized"

	def check_holiday_psd(self, date, psd_dates=None):
		"""
		Function checks if the date is a PH in Singapore or if the date is a plant shutdown date

		Parameters:
		date (datetime): The date that is inputted.
		psd_dates (list): Defaults to None. Otherwise, plant shutdown dates in the form of a list.

		Returns:
		0 if it is a PH or PSD, else 1 to represent that it is a working day
		"""
		sg_holidays = holidays.Singapore()
		if date in sg_holidays or (psd_dates is not None and date.strftime('%Y-%m-%d') in psd_dates):
			return 0
		return 1

	def filter_by_tool(self, df, tool):
		temp = df.loc[df.tool == tool]
		return temp

	def condition(self, x):
		if x == 1:
		    return 0.5
		elif x == 2 or x == 3: 
		    return 1
		else:
		    return 0

	def safe_divide(self, a, b):
		try:
			res = a / b
		except ZeroDivisionError:
			res = 0
		return res

	def parse_files(self):
		dfs = []
		for file, label in self.tool_files.items():
			with open(file) as f:
				lines = f.readlines()
				data = []
				for line in lines:
					date, name, project = line.strip().split(" - ")[0], line.strip().split(" - ")[1][1:], line.strip().split(" - ")[2][1:]
					data.append({"date/timezone": date, "analyst": name, "project": project, "tool": label})
				df = pd.DataFrame(data)
				dfs.append(df)
		self.tools_df = pd.concat(dfs, ignore_index=True)

	def preprocess(self):
		temp = self.tools_df.copy()
		temp['date'] = pd.to_datetime(temp['date/timezone'].apply(lambda x: x[:10]))
		temp['timezone'] = temp['date/timezone'].apply(lambda x: x[11:13])
		temp['analyst_cleaned'] = temp['analyst'].apply(lambda x: self.clean_analyst(x.lower()))
		temp['project_cleaned'] = temp['project'].apply(lambda x: self.clean_project(x, self.NOT_UTILIZED_WORDS))
		temp = temp.drop('date/timezone', axis=1)
		return temp

	def format_util_rate_weekly(self):
		tools = self.processed_df.tool.unique()
		dfs = []
	    
		for tool in tools:
			tool_df = self.filter_by_tool(self.processed_df, tool)
			util = tool_df.groupby(['date', 'tool', 'project_cleaned']).aggregate({'timezone': lambda x: x.nunique()}).reset_index()
			util = util.drop_duplicates(subset=['date'], keep='first')
			util = util.set_index('date').resample('D').asfreq().fillna(value={'timezone': 0, 'tool': tool, 'project_cleaned': 'tool_not_utilized'}).reset_index()
			util['is_ph_psd'] = util['date'].apply(lambda x: self.check_holiday_psd(x))
			util['rate'] = util['timezone'].apply(self.condition)
			util['work_week'] = util['date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-W%U'))

			rate = util.groupby(['work_week', 'tool']).aggregate({'is_ph_psd': lambda x: x.sum(), 'rate': lambda x: x.sum()}).reset_index()
			rate['final_rate'] = self.safe_divide(rate.rate, rate.is_ph_psd)
			dfs.append(rate)

		return pd.concat(dfs)