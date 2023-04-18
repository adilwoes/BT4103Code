import pandas as pd
import numpy as np
from datetime import datetime
from ipywidgets import Button, Layout, HBox, VBox, Tab, HTML, Dropdown, DatePicker, Output
import ipywidgets as widgets
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output, display
from codes.process_automation import AnomalyAutomation

pd.set_option('display.max_columns', 5) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', 300) #replace n with the number of rows you want to see completely
pd.set_option('display.max_colwidth', None)

class Anomaly_Detection:
    """
    A class for visualizing the techniques anomaly detection rate data 
    of various failures, technology nodes and product names.

    Parameters:
    -----------
    job_input: str
        The file path of the Job Input Form Excel file.

    TITLE_1: str
        A constant string representing the title of the techniques anomaly detection rate visualization.
    df: pandas.DataFrame
        A DataFrame containing the processed and cleaned Job Input Form DataFrame. 
    output1: widgets.Output
        An Output widget used to display the visualization.
    dropdown_product_name: widgets.Dropdown
        A dropdown widget used to select the product name for the visualization.
    dropdown_failure: widgets.Dropdown
        A dropdown widget used to select the failure for the visualization.
    dropdown_tech_node: widgets.Dropdown
        A dropdown widget used to select the tech node for the visualization.

    df_output: widgets.VBox
        The vertical box widget for displaying the dataframe.
    input_widgets: widget.HBox
        The vertical box widget for displaying the product name, failure and tech node filters.
    tab: widget.VBox
        The vertical box widget containing the title, filters widgets and dataframe widgets.
    common_filtering: customized function
        Function to help filter for product name, failure and tech node.
    """
    
    TITLE_1 = HTML(value="<h1><b>Technology Anomaly Detection Rate</b></h1>")  
        
    def __init__(self, file_path): 
        self.df = AnomalyAutomation(file_path).df

        self.dropdown_product_name = Dropdown(
            options=AnomalyAutomation(file_path).product, 
            description='Product: ')
        
        self.dropdown_failure = Dropdown(
            options=AnomalyAutomation(file_path).failure, 
            description='Failure: ')

        self.dropdown_tech_node = Dropdown(
            options=AnomalyAutomation(file_path).tech_node, 
            description='Technology Node: ')

        self.output1 = widgets.Output()

        self.dropdown_product_name.observe(self.dropdown_product_eventhandler, names='value')
        self.dropdown_failure.observe(self.dropdown_failure_eventhandler, names='value')
        self.dropdown_tech_node.observe(self.dropdown_tech_node_eventhandler, names='value')

        self.df_output = VBox([self.output1])
        item_layout = widgets.Layout(margin='20px 0 30px 40px')
        self.input_widgets = widgets.HBox([self.dropdown_product_name, self.dropdown_failure, self.dropdown_tech_node], layout=item_layout)
        self.tab = VBox([self.TITLE_1, self.input_widgets, self.df_output])
        
    def unique_sorted_values_plus_ALL(self, array):
        unique = array.unique().tolist()
        unique.sort()
        unique.insert(0, ALL)
        return unique

    def common_filtering(self, product_name, failure, tech_node):
        self.output1.clear_output()

        # 3 ALL Selection
        if (product_name == 'ALL') & (failure == 'ALL') & (tech_node == 'ALL'):
            common_filter = self.df

        # 2 ALL Selection
        elif (product_name == 'ALL') & (failure == 'ALL') & (tech_node != 'ALL') :
            common_filter = self.df[self.df.technology_node == tech_node]
        elif (product_name == 'ALL') & (failure != 'ALL') & (tech_node == 'ALL'):
            common_filter = self.df[self.df.failure == failure]
        elif (product_name != 'ALL') & (failure == 'ALL') & (tech_node == 'ALL'):
            common_filter = self.df[self.df.product_name == product_name]

        # 1 ALL Selection
        elif (product_name == 'ALL') & (failure != 'ALL') & (tech_node != 'ALL') :
            common_filter = self.df[(self.df.failure == failure) & (self.df.technology_node == tech_node)]
        elif (product_name != 'ALL') & (failure == 'ALL') & (tech_node != 'ALL'):
            common_filter = self.df[(self.df.product_name == product_name) & (self.df.technology_node == tech_node)]
        elif (product_name != 'ALL') & (failure != 'ALL') & (tech_node == 'ALL'):
            common_filter = self.df[(self.df.product_name == product_name) & (self.df.technology_node == tech_node)]

        else:
            common_filter = self.df[(self.df.product_name == product_name) & 
                                      (self.df.failure == failure) & (self.df.technology_node == tech_node)]

        with self.output1:
            display(common_filter)

    def dropdown_product_eventhandler(self, change):
        self.common_filtering(change.new, self.dropdown_failure.value, self.dropdown_tech_node.value)

    def dropdown_failure_eventhandler(self, change):
        self.common_filtering(self.dropdown_product_name.value, change.new, self.dropdown_tech_node.value)

    def dropdown_tech_node_eventhandler(self, change):
        self.common_filtering(self.dropdown_product_name.value, self.dropdown_failure.value, change.new)            
## ------------------------------ Ends here            