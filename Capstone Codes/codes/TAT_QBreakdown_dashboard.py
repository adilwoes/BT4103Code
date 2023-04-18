import re
import os
import copy
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

import ipywidgets as widgets
from ipywidgets import Button, Layout, HBox, \
        VBox, Tab, HTML, Dropdown, SelectMultiple
import plotly.graph_objects as go

from codes.process_automation import TATAutomation

pd.set_option('display.max_columns', 15) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', 50) #replace n with the number of rows you want to see completely
pd.set_option('display.max_colwidth', None)

class TatQBreakdown:
    	"""
	A class for creating interactive visualizations of KPI and KPI (Others) Tab.
    
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
	title1: str
        A constant string representing the title of the turnaround time visualization.
    title2: str
        A constant string representing the title of the high priority queue breakdown visualization.
	title3: str
        A constant string representing the title of the Overall Analysis Status visualization.
    title4: str
        A constant string representing the title of the Type Breakdown visualization.
	title5: str
        A constant string representing the title of the Type Breakdown by Month visualization.
    title6: str
        A constant string representing the title of the Product Breakdown visualization.        
    param: str
        A constant string representing the titles of the filters.
    ww_table: str
        A constant string representing the titles of the reference table.        
    br: str
        A constant string representing the break spacing.          
	            
    ww : pandas.DataFrame
        The work week calendar data.        
    df : pandas.DataFrame
        The cleaned, imputed, calculated DataFrame.           
    cancelled : pandas.DataFrame
        The DataFrame comprising all cancelled jobs.          
    priority : List
        The List of priority values.          
    type : List
        The List of type values.       
    product : List
        The List of product values.
    description : List
        The List of description values.
    workyear : List
        The List of work year values.
    workweek : List
        The List of work week values.  
        
	type_widget : widgets.Dropdown
		A dropdown widget for selecting the test type.
	product_widget : widgets.Dropdown
		A dropdown widget for selecting the product type.        
	des_widget : widgets.Dropdown
		A dropdown widget for selecting the description type.
	ww_year_start_widget : widgets.Dropdown
		A dropdown widget for selecting the start year.        
	ww_year_end_widget : widgets.Dropdown
		A dropdown widget for selecting the end year.        
	ww_start_widget : widgets.Dropdown
		A dropdown widget for selecting the start week.
	ww_end_widget : widgets.Dropdown
		A dropdown widget for selecting the end week.    
        
	refresh_btn1 : widgets.Button
		A button widget for executing the filter.        
	type_widget1 : widgets.Dropdown
		A dropdown widget for selecting the test type.
	product_widget1 : widgets.Dropdown
		A dropdown widget for selecting the product type.        
	des_widget1 : widgets.Dropdown
		A dropdown widget for selecting the description type.
	ww_year_start_widget1 : widgets.Dropdown
		A dropdown widget for selecting the start year.        
	ww_year_end_widget1 : widgets.Dropdown
		A dropdown widget for selecting the end year.        
	ww_start_widget1 : widgets.Slider
		A Slider widget for selecting the start week.
	ww_end_widget1 : widgets.Slider
		A Slider widget for selecting the end week.        
	refresh_btn1 : widgets.Button
		A button widget for executing the filter.
        
	tat_btn : widgets.Button
		A button widget for executing the reading of TAT Graph data.         
        
	turnaround_graph: plotly.graph_objs._figurewidget.FigureWidget
		A FigureWidget object for the turnaround time plot.
	csc_graph: plotly.graph_objs._figurewidget.FigureWidget
   		A FigureWidget object for the overall analysis status plot.
	priority_graph: plotly.graph_objs._figurewidget.FigureWidget
		A FigureWidget object for the high priority queue breakdown plot.
	type_graph: plotly.graph_objs._figurewidget.FigureWidget
		A FigureWidget object for the type analysis plot.
	prod_graph: plotly.graph_objs._figurewidget.FigureWidget
		A FigureWidget object for the product analysis plot.
        
	var_tab1: widgets.VBox
		A vertical box containing the filters for KPI Tab.
	var_tab2: widgets.VBox
		A vertical box containing the filters for KPI (Others) Tab.
        
	kpi_tab: widgets.Tab
		A Tab box containing the KPI Tab.
	kpi_others_tab: widgets.Tab
		A Tab box containing the KPI (Others) Tab.

	"""
    br = HTML(value="<br></br>")
    ww_table = HTML(value="<h3><b>Reference Table of Monthly Start Weeks</b></h3>")
    param = HTML(value="<h3><b>Filter According To:</b></h3>")
    title1 = HTML(value="<h1><b>Turnaround Time</b></h1>")
    title2 = HTML(value="<h1><b>High Priority Queue Time Breakdown</b></h1>")
    title3 = HTML(value="<h1><b>Overall Analysis Status</b></h1>")
    title4 = HTML(value="<h1><b>Type Breakdown</b></h1>")
    title5 = HTML(value="<h1><b>Type Breakdown by Month</b></h1>")
    title6 = HTML(value="<h1><b>Product Breakdown</b></h1>")
    
    def __init__(self, file_path, file_path_cancelled, fp_ww):
        data = TATAutomation(file_path, file_path_cancelled, fp_ww)
        self.df = data.df
        self.cancelled = data.cancelled
        self.ww = data.ww
        self.type = data.type
        self.description = data.description
        self.priority = data.priority
        self.product = data.product
        self.workyear = data.workyear
        self.workweek = data.workweek
            
        self.type_widget = SelectMultiple(options=self.type, value=['ALL'], \
                                     description='Type:',disabled=False, style={'description_width': 'initial'})

        self.type_widget1 = SelectMultiple(options=self.type, value=['ALL'], \
                                     description='Type:',disabled=False, style={'description_width': 'initial'})
        
        self.priority_widget = SelectMultiple(options=self.priority, value=['ALL'],\
                                              description='Priority: ', disabled=False, style={'description_width': 'initial'})

        self.priority_widget1 = SelectMultiple(options=self.priority, value=['ALL'],\
                                              description='Priority: ', disabled=False, style={'description_width': 'initial'})
              
        self.product_widget = SelectMultiple(options=self.product,value=['ALL'],\
                                             description='Product:', disabled=False, style={'description_width': 'initial'})

        self.product_widget1 = SelectMultiple(options=self.product,value=['ALL'],\
                                             description='Product:', disabled=False, style={'description_width': 'initial'})
              
        self.des_widget = SelectMultiple(options=self.description, value=['ALL'],\
                                        description='Job Description:', disabled=False, style={'description_width': 'initial'})

        self.des_widget1 = SelectMultiple(options=self.description, value=['ALL'],\
                                        description='Job Description:', disabled=False, style={'description_width': 'initial'})
              
        self.ww_year_start_widget = widgets.Dropdown(options=self.workyear,value=self.workyear[0],\
                                                     description='Work Year (Start)', disabled=False, \
                                                     style={'description_width': 'initial'})

        self.ww_year_start_widget1 = widgets.Dropdown(options=self.workyear, value=self.workyear[0],\
                                                     description='Work Year (Start)', disabled=False, \
                                                     style={'description_width': 'initial'})
        
        self.ww_year_end_widget = widgets.Dropdown(options=self.workyear, value=self.workyear[-1], 
                                                   description='Work Year (End)', disabled=False, \
                                                   style={'description_width': 'initial'})
              
        self.ww_year_end_widget1 = widgets.Dropdown(options=self.workyear, value=self.workyear[-1], \
                                                    description='Work Year (End)',\
                                                   disabled=False, style={'description_width': 'initial'})
        
        self.ww_start_widget = widgets.IntSlider(value=1, min=1, max=max(self.workweek), step=1,\
                                                 description='Work Week (Start):', 
                                                 disabled=False, style={'description_width': 'initial'})
        
        self.ww_start_widget1 = widgets.IntSlider(value=1, min=1,max=max(self.workweek), step=1,\
                                                 description='Work Week (Start):', disabled=False, \
                                                 style={'description_width': 'initial'})
              
        self.ww_end_widget = widgets.IntSlider(value=max(self.workweek), min=1, max=max(self.workweek), step=1, \
                                               description='Work Week (End):', disabled=False, \
                                               style={'description_width': 'initial'})

        self.ww_end_widget1 = widgets.IntSlider(value=max(self.workweek), min=1, max=max(self.workweek), step=1, \
                                               description='Work Week (End):', disabled=False, \
                                               style={'description_width': 'initial'})
        
        self.tat_btn = Button(description='Read TAT Data from Excel', button_style='primary',\
                              style={'description_width': 'initial'}, layout = Layout(width='auto'))
        self.tat_btn.on_click(self.read_tat_data)     
        self.refresh_btn = Button(description='Refresh',button_style='success')
        self.refresh_btn1 = Button(description='Refresh',button_style='success')
        self.refresh_btn.on_click(self.refresh_tab1)
        self.refresh_btn1.on_click(self.refresh_tab2)
        
        self.output1 = widgets.Output()
        with self.output1:
            display(self.ww)
            
        self.turnaround_graph = self.get_cycle_time_graph(self.df)
        self.csc_graph = self.get_monthly_completed_submitted(self.df, self.cancelled)
        self.priority_graph = self.high_priority_analysis(self.df)
        self.type_graph, self.type_month_graph = self.type_analysis(self.df)
        self.prod_graph = self.product_loading(self.df)
        
        self.var_tab1 = VBox([HBox([self.ww_year_start_widget, self.ww_start_widget]),\
                              HBox([self.ww_year_end_widget,self.ww_end_widget]), self.br, \
                    HBox([self.type_widget, self.product_widget, self.des_widget, self.priority_widget]), self.br, \
                       self.refresh_btn, self.br])

        self.var_tab2 = VBox([HBox([self.ww_year_start_widget1, self.ww_start_widget1]),\
                              HBox([self.ww_year_end_widget1, self.ww_end_widget1]), self.br, \
                        HBox([self.type_widget1, self.product_widget1, self.des_widget1, self.priority_widget1]),\
                              self.br, self.refresh_btn1, self.br])
              
        self.kpi_tab = VBox([self.ww_table, self.output1, self.param, self.var_tab1, self.title1, \
                             self.turnaround_graph, self.tat_btn, self.title2, self.priority_graph, \
                                    self.title3, self.csc_graph])

        self.kpi_others_tab = VBox([self.ww_table, self.output1, self.param, self.var_tab2, \
                              self.title4, self.type_graph,\
                              self.title5, self.type_month_graph,\
                              self.title6, self.prod_graph])
        
    def get_month_end_demand(self, df):
  		"""
        Plots the Turnaround Time graph.

        Parameters:
        -----------
        df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.       
        
        Returns:
        --------
        month_end_dd : pandas.DataFrame
            A DataFrame consisting of the aggregated month end deman values.

        """ 
        earliest_d = df['LMS Submission Date'].min()
        if df['LMS Submission Date'].max() > df['FI End'].max():
            latest_d = df['LMS Submission Date'].max()
        else:
            latest_d = df['FI End'].max()
        r = relativedelta(latest_d, earliest_d)
        months = r.months + (12*r.years)
        open_jobs_df = df.groupby(['LMS #']).first()
        month_end_dd = pd.DataFrame()
        if pd.notnull(months):
            for i in range(1,months+3):
                start_d = datetime(earliest_d.year, earliest_d.month,1) - relativedelta(days=1) + relativedelta(months=i-1)
                curr = open_jobs_df[(open_jobs_df['FI End'] > start_d) | (pd.isnull(open_jobs_df['FI End']))]
                curr = curr[curr['LMS Submission Date'] <= start_d]
                month_end_dd.loc[start_d.strftime('%Y-%m'),'Count'] = len(curr)
        else:
            month_end_dd['Count'] = 0
        return month_end_dd

    def export_cycle_time_data(self, analysis, queue, fi_end, tester_util):
  		"""
        Export the Turnaround Time Data into Excel.

        Parameters:
        -----------
        analysis : List
            A list of analysis time values.       
        
        queue : List
            A list of queue time values. 
        
        fi_end : List
            A list of FI End date values. 
        
        tester_util : List
            A list of None values.             
        """    
        df = pd.DataFrame()
        df['FI End'] = fi_end
        df['Analysis'] = analysis
        df['Queue'] = queue
        df['Tester Utilisation'] = tester_util
        df.to_excel('Data/TAT Graph Data.xlsx', index=False)
              
    #turnaround time
    def get_cycle_time_graph(self, df):
  		"""
        Plots the Turnaround Time graph.

        Parameters:
        -----------
        df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.       
        
        Returns:
        --------
        turnaround_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the Turnaround Time plot.

        """           
        df = df.groupby('LMS #').last().loc[:,['Queue', 'Analysis', 'FI End']]
        df_res = df.groupby(df['FI End'].dt.strftime('%Y-%m')).mean().reset_index()
        self.export_cycle_time_data(list(df_res['Analysis']), list(df_res['Queue']), list(df_res['FI End']), \
                               [0] * len(df_res['FI End']))
              
        layout = go.Layout(title='TAT', xaxis= dict(title='Month'),template = 'simple_white',\
                             yaxis=dict(title='# of Days'), barmode="relative", \
                                       yaxis2=dict(title=dict(text="Tester Utilisation %"),\
                                            side="right",range=[0, 100], dtick = 10, overlaying="y"),\
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        turnaround_graph = go.FigureWidget(layout=layout)
        
        turnaround_graph.add_bar(y=df_res['Analysis'], x = df_res['FI End'], name='Analysis', \
                                 marker_color = '#104E8B', yaxis='y1')
        
        turnaround_graph.add_bar(y=df_res['Queue'], x = df_res['FI End'], name = 'Queue', marker_color = '#8B1A1A', yaxis='y1')
        
        turnaround_graph.add_scatter(y=[10] * len(df_res['Analysis']), x = df_res['FI End'],\
                                     name= 'KPI Goal', line=dict(color='#A2CD5A', width=3), mode='lines',yaxis='y1')
        
        turnaround_graph.add_scatter(y=[0] * len(df_res['Analysis']), x = df_res['FI End'],\
                                    name= 'Tester Utilisation', line=dict(color='#68228B', width=3),mode='lines', yaxis="y2")
        
        return turnaround_graph

    def get_monthly_completed_submitted(self, df, cancelled):
  		"""
        Plots the Overall Analysis Status graph.

        Parameters:
        -----------
        df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.
            
        cancelled : pandas.DataFrame
            A pandas DataFrame of the Canceled data used.       
        
        Returns:
        --------
        csc_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the Overall Analysis Status plot.

        """          
        #jobs completed, submitted, cancelled
        cancelled = cancelled.groupby('LMS #').max().loc[:,['LMS Submission Date']]
        completed = df.groupby('LMS #').last().loc[:,['LMS Submission Date','FI End']]
        completedd = completed.loc[:,['FI End']].groupby(completed['FI End'].dt.strftime('%Y-%m'))\
                            .count()\
                            .rename(columns={'FI End': 'Count'})\
                            .reset_index()

        submitted = completed.loc[:,['LMS Submission Date']].groupby(completed['LMS Submission Date'].dt.strftime('%Y-%m'))\
                            .count()\
                            .rename(columns={'LMS Submission Date': 'Count'})\
                            .reset_index()
        submitted = submitted[submitted['LMS Submission Date'] <= completedd['FI End'].max()]
        cancelledd = cancelled.groupby(cancelled['LMS Submission Date'].dt.strftime('%Y-%m'))\
                        .count()\
                        .rename(columns={'LMS Submission Date': 'Count'})\
                        .reset_index()
        
        month_end_dem = self.get_month_end_demand(df)
        month_end_dem = month_end_dem[month_end_dem.index <= completedd['FI End'].max()]
        layout = go.Layout(title='Overall Analysis Status', xaxis_title='Month', \
                           yaxis_title='# of Jobs',template = 'plotly_white',\
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        csc_graph = go.FigureWidget(layout=layout)
        csc_graph.add_bar(y=submitted['Count'], x = submitted['LMS Submission Date'], marker_color = '#008000',\
                          name='Submitted')
        csc_graph.add_bar(y=completedd['Count'], x = completedd['FI End'], name='Completed', marker_color = '#D2691E')
        csc_graph.add_bar(y=cancelledd['Count'], x = cancelledd['LMS Submission Date'], name='Cancelled', marker_color='#7AC5CD')
        csc_graph.add_scatter(y=month_end_dem['Count'], x = month_end_dem.index, name='Month End Demand', mode='lines')
        return csc_graph

    def high_priority_analysis(self, df):
 		"""
        Plots the Priority Queue Breakdown graph.

        Parameters:
        -----------
        df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.
        
        Returns:
        --------
        priority_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the Priority Queue Breakdown plot.

        """        
        #high priority analysis
        df = df[['Priority #', 'LMS #', 'FI End', 'STATUS']]
        df = df.drop_duplicates()
        high_priority_df = df[df['Priority #'] == 'H']
        temp = pd.DataFrame(high_priority_df['STATUS'].value_counts())
        values = [x for x in list(df['STATUS'].unique()) if ('HOLD' in x) | ('QUEUE' in x)]
        temp = temp[temp.index.isin(values)].reset_index()

        layout = go.Layout(title='High Priority Queue Breakdown', template = 'plotly_white')
        priority_graph = go.FigureWidget(layout=layout)
        priority_graph.add_trace(go.Pie(labels=temp['index'], values=temp['STATUS']))
        return priority_graph

    def type_analysis(self, df):
 		"""
        Plots the Type analysis graph.

        Parameters:
        -----------
        df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.
        
        Returns:
        --------
        type_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the type analysis plot.

        type_month_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the type analysis by month plot.
        """
        df = df[['LMS #','TYPE','FI End']]
        df = df.drop_duplicates()
        type_df = df['TYPE'].value_counts().reset_index()
        layout = go.Layout(title='SGP Analysis Type', template = 'plotly_white')
        type_graph = go.FigureWidget(layout=layout)
        type_graph.add_trace(go.Pie(labels=type_df['index'], values=type_df['TYPE']))

        type_month = df.groupby(['TYPE', df['FI End'].dt.strftime('%Y-%m')])\
                            .count()\
                            .rename(columns={'FI End': 'Count'})\
                            .reset_index()

        layout = go.Layout(title='Type Analysis By Month', xaxis_title='Month', yaxis_title='# of Usage',\
                           template = 'plotly_white')
        type_month_graph = go.FigureWidget(layout=layout)
        type_month_graph.add_bar(y=type_month['Count'], x = type_month['FI End'], text = type_month["TYPE"], \
                                 name='Type', base='stack')
        return type_graph, type_month_graph

    def product_loading(self, df):
 		"""
        Plots the product loading graph.

        Parameters:
        -----------
        df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.
        
        Returns:
        --------
        prod_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the product loading plot.         
        """
        #product loading
        df = df[['LMS #','Product', 'FI End']]
        df = df.drop_duplicates()
        prod = df['Product'].value_counts().reset_index()
        layout = go.Layout(title='SGP Product Loading', template = 'plotly_white')
        prod_graph = go.FigureWidget(layout=layout)
        prod_graph.add_trace(go.Pie(labels=prod['index'], values=prod['Product']))
        return prod_graph

## ------------------------------ FILTER FUNCTIONS                
    def filter_data1(self, graph_df):
		"""
        Filters data according filters set.

        Parameters:
        -----------
        graph_df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.
        
        Returns:
        --------
        graph_df : pandas.DataFrame
            A filtered pandas DataFrame of the Calculated data used.            

        """          
        if 'ALL' not in self.type_widget1.value:
            not_in = list(set(self.type) - set(self.type_widget1.value))
            for val in not_in:
                graph_df = graph_df[graph_df['TYPE'] != val]

        if 'ALL' not in self.product_widget1.value:
            not_in = list(set(self.product) - set(self.product_widget1.value))
            for val in not_in:
                graph_df = graph_df[graph_df['Product'] != val]

        if 'ALL' not in self.des_widget1.value:
            not_in = list(set(self.description) - set(self.des_widget1.value))
            for val in not_in:
                graph_df = graph_df[graph_df['JOB DESCRIPTION'] != val]

        if 'ALL' not in self.priority_widget1.value:
            not_in = list(set(self.priority) - set(self.priority_widget1.value))
            for val in not_in:
                graph_df = graph_df[graph_df['Priority #'] != val]

        graph_df = graph_df[(graph_df['FI End Week'] >= self.ww_start_widget1.value) & \
                            (graph_df['FI End Year'] >= self.ww_year_start_widget1.value)]

        graph_df = graph_df[(graph_df['FI End Week'] <= self.ww_end_widget1.value) & \
                        (graph_df['FI End Year'] <= self.ww_year_end_widget1.value)]
        
        return graph_df
    
    def filter_data(self, graph_df):
		"""
        Filters data according filters set.

        Parameters:
        -----------
        graph_df : pandas.DataFrame
            A pandas DataFrame of the Calculated data used.
        
        Returns:
        --------
        graph_df : pandas.DataFrame
            A filtered pandas DataFrame of the Calculated data used.            

        """        
        if 'ALL' not in self.type_widget.value:
            not_in = list(set(self.type) - set(self.type_widget.value))
            for val in not_in:
                graph_df = graph_df[graph_df['TYPE'] != val]

        if 'ALL' not in self.product_widget.value:
            not_in = list(set(self.product) - set(self.product_widget.value))
            for val in not_in:
                graph_df = graph_df[graph_df['Product'] != val]

        if 'ALL' not in self.des_widget.value:
            not_in = list(set(self.description) - set(self.des_widget.value))
            for val in not_in:
                graph_df = graph_df[graph_df['JOB DESCRIPTION'] != val]

        if 'ALL' not in self.priority_widget.value:
            not_in = list(set(self.priority) - set(self.priority_widget.value))
            for val in not_in:
                graph_df = graph_df[graph_df['Priority #'] != val]

        graph_df = graph_df[~(graph_df['FI End Year'] < self.ww_year_start_widget.value)]
        graph_df = graph_df[~(graph_df['FI End Year'] > self.ww_year_end_widget.value)]
        
        graph_df = graph_df[~((graph_df['FI End Week'] <= self.ww_start_widget.value) & \
                            (graph_df['FI End Year'] == self.ww_year_start_widget.value))]

        graph_df = graph_df[~((graph_df['FI End Week'] >= self.ww_end_widget.value) & \
                        (graph_df['FI End Year'] == self.ww_year_end_widget.value))]
        return graph_df
## ------------------------------ READ TAT TABLE FUNCTIONS
    def read_tat_data(self, evt):
		"""
        Reads the TAT Graph Data Excel and updates 'Tester Utilisation' data

        Parameters:
        -----------
        evt : event object
            Event object passed to the method.

        """
        df = pd.read_excel('Data/TAT graph data.xlsx')
        self.turnaround_graph.data[3]['y'] = df['Tester Utilisation']
              
              
## ------------------------------ REFRESH FUNCTIONS  
    def refresh_tab1(self, evt):
		"""
        Refreshes the KPI Tab.

        Parameters:
        -----------
        evt : event object
            Event object passed to the method.

        """
        graph_df = self.filter_data(self.df)
        cancelled = self.filter_data(self.cancelled)
        
        tat_data = self.get_cycle_time_graph(graph_df).data
        self.turnaround_graph.data[0]['y'] = tat_data[0]['y']
        self.turnaround_graph.data[1]['y'] = tat_data[1]['y']
        self.turnaround_graph.data[2]['y'] = tat_data[2]['y']
        self.turnaround_graph.data[3]['y'] = tat_data[3]['y']
        self.turnaround_graph.data[0]['x'] = tat_data[0]['x']
        self.turnaround_graph.data[1]['x'] = tat_data[1]['x']
        self.turnaround_graph.data[2]['x'] = tat_data[2]['x']
        self.turnaround_graph.data[3]['x'] = tat_data[3]['x']
        self.export_cycle_time_data(tat_data[0]['y'], \
                               tat_data[1]['y'], \
                               tat_data[1]['x'], tat_data[2]['y'])
        new_data = self.get_monthly_completed_submitted(graph_df, cancelled).data
        self.csc_graph.data[0]['x'] = new_data[0]['x']
        self.csc_graph.data[1]['x'] = new_data[1]['x'] 
        self.csc_graph.data[2]['x'] = new_data[2]['x']
        self.csc_graph.data[3]['x'] = new_data[3]['x']
        self.csc_graph.data[0]['y'] = new_data[0]['y']
        self.csc_graph.data[1]['y'] = new_data[1]['y'] 
        self.csc_graph.data[2]['y'] = new_data[2]['y']
        self.csc_graph.data[3]['y'] = new_data[3]['y']
        self.priority_graph.data[0].values = self.high_priority_analysis(graph_df).data[0].values
        self.priority_graph.data[0].labels = self.high_priority_analysis(graph_df).data[0].labels


    def refresh_tab2(self, evt):
		"""
        Refreshes the KPI (Others) Tab.

        Parameters:
        -----------
        evt : event object
            Event object passed to the method.

        """
        graph_df = self.filter_data1(self.df)

        self.type_graph.data[0].values = self.type_analysis(graph_df)[0].data[0].values
        self.type_graph.data[0].labels = self.type_analysis(graph_df)[0].data[0].labels
        self.type_month_graph.data[0]['text'] = self.type_analysis(graph_df)[1].data[0]['text']
        self.type_month_graph.data[0]['x'] = self.type_analysis(graph_df)[1].data[0]['x']
        self.type_month_graph.data[0]['y'] = self.type_analysis(graph_df)[1].data[0]['y']
        self.prod_graph.data[0].values = self.product_loading(graph_df).data[0].values
        self.prod_graph.data[0].labels = self.product_loading(graph_df).data[0].labels
              


              