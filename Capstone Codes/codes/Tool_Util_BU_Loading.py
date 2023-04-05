import pandas as pd
import numpy as np
from datetime import datetime
from ipywidgets import Button, Layout, HBox, VBox, Tab, HTML, Dropdown, DatePicker, Output
import ipywidgets as widgets
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output, display

from codes.process_automation import ProcessAutomation, ProcessToolUtilization

class ToolUtilBULoading:

	TITLE_1 = HTML(value="<h1><b>Technology Node Distribution</b></h1>")
	TITLE_2 = HTML(value="<h1><b>Product and Failure Distribution</b></h1>")
	TITLE_3 = HTML(value="<h1><b>SGP BU Loading</b></h1>")
	TITLE_4 = HTML(value="<h1><b>Tool Utilization Rate (Weekly)</b></h1>")
	TEMPLATE = "plotly_white"
	COLOR_CODES = ['#ff7f0e', '#1f77b4', '#FFBF5F', '#aec7e8', '#7f7f7f']

	def __init__(self, tool_files, job_input):
		self.tool_df = ProcessToolUtilization(tool_files).final_df
		self.auto_df = ProcessAutomation(job_input).df

		## Tech Node, Product and Failure Distribution Viz
		self.tech_node_options = list(self.auto_df['technology node'].unique())
		self.tech_node_dropdown = Dropdown(
		    options=self.tech_node_options, 
		    value=self.tech_node_options[0], 
		    description='Technology Node:', style={'description_width': 'initial'})

		self.prod_name_options = list(self.auto_df[self.auto_df['technology node']==self.tech_node_dropdown.value]['product name'].unique())
		self.prod_name_dropdown = Dropdown(
		    options=self.prod_name_options, 
		    value=self.prod_name_options[0], 
		    description='Product Name:', style={'description_width': 'initial'})

		self.prod_refresh_btn = Button(
		    description='Refresh',
		    button_style='success', style={'description_width': 'initial'}
		)

		self.fail_refresh_btn = Button(
		    description='Refresh',
		    button_style='success', style={'description_width': 'initial'}
		)

		self.tn_freq_graph = self.plot_tn_graph()
		self.prod_freq_graph = self.plot_prod_graph(self.tech_node_dropdown.value)
		self.fail_freq_graph = self.plot_fail_graph(self.tech_node_dropdown.value, self.prod_name_dropdown.value)

		self.tech_node_dropdown.observe(self.update_prod_name_dropdown, 'value')
		self.prod_refresh_btn.on_click(self.refresh_prod_graph)
		self.fail_refresh_btn.on_click(self.refresh_fail_graph)

		self.prod_graph = VBox([HBox([self.tech_node_dropdown, self.prod_refresh_btn]), self.prod_freq_graph])
		self.fail_graph = VBox([HBox([self.prod_name_dropdown, self.fail_refresh_btn]), self.fail_freq_graph])
		self.graph1 = VBox([self.tn_freq_graph]) 
		self.graph2 = HBox([self.prod_graph, self.fail_graph])
		self.tab = VBox([self.TITLE_1, self.graph1, self.TITLE_2, self.graph2])

		## ------------------------------ Ends here

		##BU Loading Stacked Barchart Viz
		self.output1 = widgets.Output()

		self.all_months = sorted(self.auto_df.year_month.unique())
		self.start_month_dropdown = Dropdown(
			options=self.all_months, 
			description='Start Month:', style={'description_width': 'initial'})
		self.end_month_dropdown = Dropdown(
			options=self.all_months, 
			value=self.all_months[-1], 
			description='End Month:', style={'description_width': 'initial'})

		self.start_month_dropdown.observe(self.on_month_filter_change, names='value')
		self.end_month_dropdown.observe(self.on_month_filter_change, names='value')

		self.filtered_bu_df = self.auto_df[(self.auto_df['year_month'] >= self.start_month_dropdown.value) & (self.auto_df['year_month'] <= self.end_month_dropdown.value)]
		self.stacked_barchart = self.plot_bu_barchart(self.filtered_bu_df)

		with self.output1:
			display(self.stacked_barchart)

		self.bu_graph = VBox([self.output1])
		self.month_selector = HBox([self.start_month_dropdown, self.end_month_dropdown])
		self.tab2 = VBox([self.TITLE_3, self.month_selector, self.stacked_barchart])

		## ------------------------------ Ends here

		##Tool Utilization Viz
		self.output2 = widgets.Output()

		self.all_weeks = sorted(self.tool_df.work_week.unique())
		self.start_week_dropdown = Dropdown(
			options=self.all_weeks, 
			description='Start Week:', style={'description_width': 'initial'})
		self.end_week_dropdown = Dropdown(
			value=self.all_weeks[-1], 
			options=self.all_weeks, 
			description='End Week:', style={'description_width': 'initial'})

		self.start_week_dropdown.observe(self.on_week_filter_change, names='value')
		self.end_week_dropdown.observe(self.on_week_filter_change, names='value')

		self.filtered_tool_df = self.tool_df[(self.tool_df['work_week'] >= self.start_week_dropdown.value) & (self.tool_df['work_week'] <= self.end_week_dropdown.value)]
		self.util_chart = self.plot_weekly_util_rate(self.filtered_tool_df)

		with self.output2:
			display(self.util_chart)

		self.util_graph = VBox([self.output2])
		self.week_selector = HBox([self.start_week_dropdown, self.end_week_dropdown])
		self.tab3 = VBox([self.TITLE_4, self.week_selector, self.util_chart])

	## ------------------------------ Ends here

	## Tech Node, Product and Failure Distribution Plotting Methods -- starts here
	def plot_tn_graph(self):
		tech_node_freq = self.auto_df['technology node'].value_counts()

		layout = go.Layout(title='Frequency of Technology Nodes', template='plotly_dark')
		tn_freq_graph = go.FigureWidget(layout=layout)
		tn_freq_graph.add_bar(y=tech_node_freq.values, x=tech_node_freq.index)

		return tn_freq_graph

	def plot_prod_graph(self, tech_node):
		prod_name_freq = self.auto_df[self.auto_df['technology node']==tech_node]['product name'].value_counts()

		layout = go.Layout(title=f'Frequency of Product Names for {tech_node}', template='plotly_dark')
		prod_freq_graph = go.FigureWidget(layout=layout)
		prod_freq_graph.add_bar(y=prod_name_freq.values, x=prod_name_freq.index)
    
		return prod_freq_graph

	def plot_fail_graph(self, tech_node, prod_name):
		failure_freq = self.auto_df[(self.auto_df['technology node']==tech_node) & (self.auto_df['product name']==prod_name)]['failure'].value_counts()

		layout = go.Layout(title=f'Frequency of Failure for {prod_name} in {tech_node}', template='plotly_dark')
		fail_freq_graph = go.FigureWidget(layout=layout)
		fail_freq_graph.add_bar(y=failure_freq.values, x=failure_freq.index)

		return fail_freq_graph

	def update_prod_name_dropdown(self, change):
		tech_node_value = self.tech_node_dropdown.value
		new_prod_name_options = list(self.auto_df[self.auto_df['technology node']==tech_node_value]['product name'].unique())
		self.prod_name_dropdown.options = new_prod_name_options
		self.prod_name_dropdown.value = new_prod_name_options[0]

	def refresh_prod_graph(self, evt):
		graph_df = self.auto_df.copy()
		prod_freq_graph_new = self.plot_prod_graph(self.tech_node_dropdown.value)
		self.prod_freq_graph.data[0]['x'] = prod_freq_graph_new.data[0]['x']
		self.prod_freq_graph.data[0]['y'] = prod_freq_graph_new.data[0]['y']
		self.prod_freq_graph.update_layout(title_text=f'Frequency of Product Names for {self.tech_node_dropdown.value}')

	def refresh_fail_graph(self, evt):
		graph_df = self.auto_df.copy()
		fail_freq_graph_new = self.plot_fail_graph(self.tech_node_dropdown.value, self.prod_name_dropdown.value)
		self.fail_freq_graph.data[0]['x'] = fail_freq_graph_new.data[0]['x']
		self.fail_freq_graph.data[0]['y'] = fail_freq_graph_new.data[0]['y']
		self.fail_freq_graph.update_layout(title_text=f'Frequency of Failure for {self.prod_name_dropdown.value} in {self.tech_node_dropdown.value}')
	
	## ------------------------------ Ends here

	## BU Loading Stacked Barchart Plotting Methods
	def plot_bu_barchart(self, df):
		temp = df.groupby(['year_month', 'bu'])['job id'].count().unstack().fillna(0).reset_index()

		fig = go.FigureWidget(data=[go.Bar(x=temp['year_month'], y=temp[col], name=col, marker=dict(color=color)) for color, col in zip(self.COLOR_CODES, temp.columns[1:])])
		fig.update_layout(barmode='stack', xaxis_title='Year-Month', yaxis_title='Job Count', title="BU Loading", template=self.TEMPLATE)

		return fig

	def on_month_filter_change(self, change):
		with self.output1:
			clear_output(wait=True)
			filtered_bu_df = self.auto_df.loc[(self.auto_df.year_month >= self.start_month_dropdown.value) & (self.auto_df.year_month <= self.end_month_dropdown.value)]
			stacked_barchart = self.plot_bu_barchart(filtered_bu_df)      
			self.stacked_barchart.data[0]['x'] = stacked_barchart.data[0]['x']
			self.stacked_barchart.data[0]['y'] = stacked_barchart.data[0]['y']
			self.stacked_barchart.data[1]['x'] = stacked_barchart.data[1]['x']
			self.stacked_barchart.data[1]['y'] = stacked_barchart.data[1]['y']
			self.stacked_barchart.data[2]['x'] = stacked_barchart.data[2]['x']
			self.stacked_barchart.data[2]['y'] = stacked_barchart.data[2]['y'] 
			self.stacked_barchart.data[3]['x'] = stacked_barchart.data[3]['x']
			self.stacked_barchart.data[3]['y'] = stacked_barchart.data[3]['y']
			self.stacked_barchart.data[4]['x'] = stacked_barchart.data[4]['x']
			self.stacked_barchart.data[4]['y'] = stacked_barchart.data[4]['y'] 
			stacked_barchart.show()

	## ------------------------------ Ends here

	## Tool Utilization Plotting Methods
	def plot_weekly_util_rate(self, df):
		layout = go.Layout(title='Tool Utilization Rate', template='plotly_dark', yaxis=dict(tickformat=".0%"))

		util_rate_graph = go.FigureWidget(layout=layout)

		for tool in df.tool.unique():
			temp_tool = df.loc[df.tool == tool]
			util_rate_graph.add_scatter(y=temp_tool.final_rate, x=temp_tool.work_week, name=tool)

		return util_rate_graph

	def on_week_filter_change(self, change):
		with self.output2:
			clear_output(wait=True)
			filtered_tool_df = self.tool_df.loc[(self.tool_df.work_week >= self.start_week_dropdown.value) & (self.tool_df.work_week <= self.end_week_dropdown.value)]
			weekly_util_graph = self.plot_weekly_util_rate(filtered_tool_df)
			self.util_chart.data[0]['x'] = weekly_util_graph.data[0]['x']
			self.util_chart.data[0]['y'] = weekly_util_graph.data[0]['y']
			self.util_chart.data[1]['x'] = weekly_util_graph.data[1]['x']
			self.util_chart.data[1]['y'] = weekly_util_graph.data[1]['y']
			self.util_chart.data[2]['x'] = weekly_util_graph.data[2]['x']
			self.util_chart.data[2]['y'] = weekly_util_graph.data[2]['y']           
			weekly_util_graph.show()

	## ------------------------------ Ends here


