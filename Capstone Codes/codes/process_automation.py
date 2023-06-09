import pandas as pd
import json
import holidays
import warnings
warnings.simplefilter("ignore")

class ProcessAutomation:
	"""
	A class for processing and cleaning data from the Job Input Form Excel file and saving it to a new Excel file.
	Parameters:
	-----------
	file_path : str
		The file path of the Job Input Form Excel file.
	
	Attributes:
	-----------
	file_path : str
		The file path of the Job Input Form Excel file.
	df : pandas.DataFrame
		The processed and cleaned DataFrame.
	"""

	def __init__(self, file_path):
		self.file_path = file_path
		self.job_input_df = pd.DataFrame()
		self.tech_anomaly_df = pd.DataFrame()
		self.process_job_input()
		self.process_tech_anomaly_detection_data()

	def process_job_input(self):
		"""
		Process the 'Hist. Data' sheet of the Job Input Form Excel file by converting column names to lowercase, 
		converting 'date finished' column to datetime format, and creating a new 'year_month' column.
	
		Parameters:
		-----------
		None
		
		Returns:
		--------
		None
		
		Raises:
		-------
		pandas.errors.EmptyDataError:
			If the 'Hist. Data' sheet is empty.
		"""
		
		self.job_input_df = pd.read_excel(self.file_path, sheet_name='Hist. Data')
		self.job_input_df['Date Finished'] = pd.to_datetime(self.job_input_df['Date Finished'])
		self.job_input_df['Year Month'] = self.job_input_df['Date Finished'].apply(lambda x: x.strftime("%Y-%m"))
	
	def process_tech_anomaly_detection_data(self):
		"""
		Processes data from tech anomaly detection.
		Takes data from tech anomaly detection and processes it to create a new DataFrame
		that shows the technique anomaly rate for each product name, failure, and technology
		node combination. This method computes the average of the techniques used by each
		combination, creates a new DataFrame that shows the technique anomaly rate for
		each combination, and returns the result.
		Returns:
		--------
		None
		"""

		columns = ["Product Name", "Failure", "Technology Node"]
		df = self.job_input_df[["Product Name", "Failure", "Technology Node", "Techniques"]]
		df = df.reset_index(drop = True)

		t = json.loads(df["Techniques"][0].replace("'", '"'))
		techniques = list(map(lambda x: json.loads(x.replace("'", '"')), df["Techniques"].tolist()))
		keys = []
		for t in techniques:
			keys.extend(list(t.keys()))
		keys = list(set(keys))

		self.tech_anomaly_df = df.copy()
		for k in keys:
			l = list(map(lambda x: (1 if x[k][0] == "Yes" else 0) if k in x.keys() else None, techniques))
			self.tech_anomaly_df[k] = l

		self.tech_anomaly_df = self.tech_anomaly_df.drop("Techniques", axis = 1)
		self.tech_anomaly_df = self.tech_anomaly_df.groupby(["Product Name", "Failure", "Technology Node"]).apply(lambda x: x.sum()/x.count()).reset_index()

		analysis = self.tech_anomaly_df[keys].to_dict()
		for analy in analysis.keys():
			analysis[analy] = list(map(lambda x: "" if str(x) == "nan" else f"{analy}: {x * 100}%, ", list(analysis[analy].values())))
		result = pd.DataFrame(analysis).apply("".join, axis = 1)

		self.tech_anomaly_df = self.tech_anomaly_df[columns]
		self.tech_anomaly_df["Technique Anomaly Rate"] = result
		self.tech_anomaly_df = self.tech_anomaly_df.reset_index()
	
	def to_excel(self):
		"""
		Save the processed and cleaned DataFrames to new Excel files in the 'Data' directory with the filenames 
		'job_input_form_powerbi.xlsx' and 'tech_anomaly_detection_rate.xlsx', respectively.
		Parameters:
		-----------
		None
		Returns:
		--------
		None
		Raises:
		-------
		IOError:
			If there is an error writing to either of the new Excel files.
		"""

		self.job_input_df.to_excel(f'Data/job_input_form_powerbi.xlsx', index=False)
		self.tech_anomaly_df.to_excel(f'Data/tech_anomaly_detection_rate.xlsx', index=False)

class TATAutomation:
    """
    A class for processing data and initialising data from the Calculated Excel file

    Parameters:
    -----------
    file_path : str
        The file path of the Singapore Device Priority - Calculated file.

    file_path_cancelled : str
        The file path of the Singapore Device Priority - Cancelled Excel file.

    ww_fp : str
        The file path of the Work Week Calendar Excel file.

    Attributes:
    -----------
    file_path : str
        The file path of the Job Input Form Excel file.

    file_path_cancelled : str
        The file path of the Singapore Device Priority - Cancelled Excel file.

    ww_fp : str
        The file path of the Work Week Calendar Excel file.

    ww : pandas.DataFrame
        The work week calendar data.

    df : pandas.DataFrame
        The cleaned, imputed, calculated DataFrame.

    cancelled : pandas.DataFrame
        The DataFrame comprising all cancelled jobs

    priority : List
        The List of priority values

    type : List
        The List of type values

    product : List
        The List of product values

    description : List
        The List of description values

    workyear : List
        The List of work year values

    workweek : List
        The List of work week values

    """
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
        """
        Process the the unique values that are available in the filter function of the dashboard.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """        
        self.df = pd.read_excel(self.file_path)
        self.cancelled = pd.read_excel(self.file_path_cancelled)
        self.priority = list(self.df['Priority #'].unique())
        self.priority.append('ALL')
        self.priority.sort()
        self.type = list(self.df['TYPE'].unique())
        self.type.append('ALL')
        self.type.sort()     
        self.description = list(self.df['JOB DESCRIPTION'].unique())
        self.description.append('ALL')
        self.description.sort()      
        self.product = list(self.df['Product'].unique())
        self.product.append('ALL')
        self.product.sort()        
        self.workyear = list(self.df['Work Year'].unique())
        self.workweek = list(self.df['Work Week'].unique())

    def process_ww(self):
        """
        Filters out applicable date period in work week calendar according to input data

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """        
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
	"""
	A class that processes tool utilization data and calculates the utilization rate.
	
	Parameters:
	-----------
	tool_files : dict
		A dictionary mapping file paths to tool labels.
	Attributes:
	-----------
	ANALYSTS: list
		A list of analyst names.
	NOT_UTILIZED_WORDS: list
		A list of words that indicate that a tool was not utilized.
	tool_files : dict
		A dictionary mapping file paths to tool labels.
	processed_df : pandas.DataFrame
		A dataframe containing the cleaned and preprocessed tool utilization data.
	final_df : pandas.DataFrame
		A dataframe containing the monthly utilization rates for each tool.
	"""

	ANALYSTS = ['david jinjie', 'hu haoran', 'nathan linarto', 
			'winson lua', 'ng kaihui', 'angeline phoa', 
			'gopinath ranganathan', 'venkat-krishnan ravikumar', 
			'seah yi-xuan', 'vasanth somasundaram', 'nicholas tee']

	NOT_UTILIZED_WORDS = ["upgrade", "pm", "repair", "test", "check"]

	def __init__(self, tool_files, psd_filepath):
		self.tool_files = tool_files
		self.psd = self.get_psd_dates(psd_filepath)
		self.tools_df = self.parse_files()
		self.processed_df = self.preprocess()
		self.final_df = self.format_util_rate_monthly()

	def is_abbreviation(self, abbr, word):
		"""
		Determine whether a string is an abbreviation of another string.
		Parameters:
		----------
		abbr : str
			The abbreviation to check.
		word : str
			The word to check if it matches the abbreviation.
		Returns:
		-------
		bool
			True if the abbreviation matches the word, False otherwise.
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
		"""
		Clean up an analyst name.
		Parameters:
		----------
		analyst_name : str
			The name of the analyst to clean up.
		Returns:
		-------
		str
			The cleaned-up name of the analyst.
		Notes:
		-----
		If the analyst name is "venkatram", it will be returned unchanged. Otherwise, the function will try to find a match
		in the list of known analyst names (self.ANALYSTS), using an abbreviation matching algorithm implemented by the
		is_abbreviation() method. If a match is found, the corresponding analyst name is returned. If no match is found,
		the original analyst name is returned unchanged.
		"""
		if analyst_name == "venkatram":
			return analyst_name
		for name in self.ANALYSTS:
			if self.is_abbreviation(analyst_name, name):
				return name
		return analyst_name

	def clean_project(self, project_details, word_list):
		"""
		Cleans the project details and returns a cleaned project description.
		Parameters:
		----------
		project_details : str
			A string representing the project details.
		word_list : List[str]
			A list of strings representing the words to check in the project details.
		Returns:
		-------
		str
			A cleaned project description. If any word in the word_list is present in the project_details, 
			returns "upgrade", otherwise returns "tool_utilized".
		"""
		if any(word in project_details.lower() for word in word_list):
			return "upgrade"
		else:
			return "tool_utilized"
		
	def get_psd_dates(self, filepath):
		"""
		Gets the Plant Shutdown Dates from the excel file "Work Week Calendar.xlsx"
		Returns:
		----------
		List
			A list of the Plant Shutdown Dates inputted in the excel file
		"""
		temp = pd.read_excel(filepath, sheet_name='Plant Shutdown Dates')
		temp = temp.iloc[:, :1]
		psd_dates = temp.Dates.to_list()
		return psd_dates
		
	def check_holiday_psd(self, date, psd_dates=None):
		"""
		Checks if the date is a PH in Singapore or if the date is a plant shutdown date
		Parameters:
		----------
		date : datetime.datetime
			Date to be checked for holiday/PSD status.
		psd_dates : list, optional
			List of plant shutdown dates, in 'YYYY-MM-DD' format. Default is None.
		Returns:
		----------
		int
			0 if the date falls on a holiday/PSD no work day, else 1.
		"""
		sg_holidays = holidays.Singapore()
		if date in sg_holidays or (psd_dates is not None and date in psd_dates):
			return 0
		return 1
	
	def filter_by_tool(self, df, tool):
		"""
		Filters a pandas DataFrame by the specified tool.
		Parameters:
		----------
		df : pandas.DataFrame
			The DataFrame to be filtered.
		tool : str
			The name of the tool to filter by.
		Returns:
		-------
		pandas.DataFrame
			A filtered DataFrame containing only the rows where the tool column matches the specified tool name.
		"""
		temp = df.loc[df.tool == tool]
		return temp
	
	def condition(self, x):
		"""
		Return a numeric value based on the input value `x`.
		Parameters:
		----------
		x : int
			An integer value representing a condition.
		Returns:
		-------
		float
			A numeric value based on the input condition:
			- If `x` is equal to 1, return 0.5.
			- If `x` is equal to 2 or 3, return 1.
			- Otherwise, return 0.
		Examples:
		--------
		>>> obj = MyClass()
		>>> obj.condition(1)
		0.5
		>>> obj.condition(2)
		1
		>>> obj.condition(4)
		0
		"""
		if x == 1:
			return 0.5
		elif x == 2 or x == 3:
			return 1
		else:
			return 0
		
	def safe_divide(self, a, b):
		"""
		Divide two numbers and return the result. If `b` is zero, return zero instead of raising an exception.
		Parameters:
		----------
		a : float
			The numerator.
		b : float
			The denominator.
		Returns:
		-------
		float
			The result of dividing `a` by `b`, or zero if `b` is zero.
		"""
		try:
			res = a / b
		except ZeroDivisionError:
			res = 0
		return res

	def parse_files(self):
		"""
		Parses the files in `tool_files` attribute and returns a concatenated dataframe.
		Returns:
		-------
		pandas.DataFrame
			A concatenated dataframe that contains the parsed data from all the files.
		"""
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
		return pd.concat(dfs, ignore_index=True)

	def preprocess(self):
		"""
		Preprocesses the data in the tools dataframe by creating new columns and dropping unnecessary columns.
		Returns:
		-------
		pandas.DataFrame
			The preprocessed tools dataframe with the following new columns:
			- 'date': a datetime object representing the date of each entry
			- 'timezone': a string representing the timezone of each entry
			- 'analyst_cleaned': a string representing the cleaned analyst name for each entry
			- 'project_cleaned': a string representing the cleaned project name for each entry
		"""
		temp = self.tools_df.copy()
		temp['date'] = pd.to_datetime(temp['date/timezone'].apply(lambda x: x[:10]))
		temp['timezone'] = temp['date/timezone'].apply(lambda x: x[11:13])
		temp['analyst_cleaned'] = temp['analyst'].apply(lambda x: self.clean_analyst(x.lower()))
		temp['project_cleaned'] = temp['project'].apply(lambda x: self.clean_project(x, self.NOT_UTILIZED_WORDS))
		temp = temp.drop('date/timezone', axis=1)
		return temp

	def format_util_rate_monthly(self):
		"""
		Format utilization rate data on a monthly basis by resampling and aggregating
		data in the processed dataframe.
		
		Returns:
		--------
		pd.DataFrame:
			Dataframe containing the formatted utilization rate data by month and tool.
			Columns include: 'work_month', 'tool', 'is_ph_psd', 'rate', and 'final_rate'.
		"""
		tools = self.processed_df.tool.unique()
		dfs = []
		
		for tool in tools:
			tool_df = self.filter_by_tool(self.processed_df, tool)
			util = tool_df.groupby(['date', 'tool', 'project_cleaned']).aggregate({'timezone': lambda x: x.nunique()}).reset_index()
			util = util.drop_duplicates(subset=['date'], keep='first')
			util = util.set_index('date').resample('D').asfreq().fillna(value={'timezone': 0, 'tool': tool, 'project_cleaned': 'tool_not_utilized'}).reset_index()
			util['is_ph_psd'] = util['date'].apply(lambda x: self.check_holiday_psd(x, self.psd))
			util['rate'] = util['timezone'].apply(self.condition)
			util.loc[util.is_ph_psd == 0, 'rate'] = 0
			util['work_month'] = util['date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m'))

			rate = util.groupby(['work_month', 'tool']).aggregate({'is_ph_psd': lambda x: x.sum(), 'rate': lambda x: x.sum()}).reset_index()
			rate['final_rate'] = self.safe_divide(rate.rate, rate.is_ph_psd)
			rate['final_rate'] = rate['final_rate'].fillna(0)
			dfs.append(rate)

		return pd.concat(dfs)
	
	def to_excel(self):
		"""
		Save the processed and cleaned tool utilisation DataFrame to new Excel files in the 'Data' directory 
		with the filenames 'tool_util.xlsx'.
		Parameters:
		-----------
		None
		Returns:
		--------
		None
		Raises:
		-------
		IOError:
			If there is an error writing to either of the new Excel files.
		"""

		self.final_df.to_excel(f'Data/tool_util.xlsx', index=False)
