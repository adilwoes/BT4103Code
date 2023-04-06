from ipywidgets import Button, HBox, VBox, HTML, Dropdown
import ipywidgets as widgets
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output, display

from codes.process_automation import ProcessAutomation

class TechNodeViz:
	"""
	A class for creating interactive visualizations of technology node, product, and failure distributions,
    as well as BU loading stacked barchart.
    
    Parameters:
    -----------
    job_input: str
        The file path of the Job Input Form Excel file.
	
	Attributes:
    -----------
	TITLE_1: str
        A constant string representing the title of the technology node and product distribution visualization.
    TITLE_2: str
        A constant string representing the title of the product and failure distribution visualization.
    TEMPLATE: str
        A constant string representing the plotly template to be used for all the visualizations.
	auto_df: pandas.DataFrame
        A DataFrame containing the processed and cleaned Job Input Form DataFrame. 
	tech_node_options: list
		A list of unique values in the 'technology node' column of the auto_df DataFrame
	tech_node_dropdown : widgets.Dropdown
		A dropdown widget for selecting the technology node.
	prod_name_options: list
		A list of unique values in the 'product name' column of the auto_df DataFrame where the 
		'technology node' column is equal to the value of the tech_node_dropdown attribute
	prod_name_dropdown: widgets.Dropdown
		A dropdown widget for selecting the product name.
	prod_refresh_btn: widgets.Button
		A button widget for refreshing the product distribution plot.
	fail_refresh_btn: widgets.Button
		A button widget for refreshing the failure distribution plot.
	tn_freq_graph: plotly.graph_objs._figurewidget.FigureWidget
		A FigureWidget object for the technology node distribution plot.
	prod_freq_graph: plotly.graph_objs._figurewidget.FigureWidget
   		A FigureWidget object for the product distribution plot.
	fail_freq_graph: plotly.graph_objs._figurewidget.FigureWidget
		A FigureWidget object for the failure distribution plot.
	prod_graph: widgets.VBox
		A vertical box containing the technology node dropdown and the product distribution plot.
	fail_graph: widgets.VBox
		A vertical box containing the product dropdown and the failure distribution plot.
	graph1: widgets.VBox
		A vertical box containing the technology node distribution plot.
	graph2: widgets.HBox
		A horizontal box containing the product and failure distribution plots.
	tab: widgets.VBox
		A vertical box containing the title and all visualizations.
	"""

	TITLE_1 = HTML(value="<h1><b>Technology Node Distribution</b></h1>")
	TITLE_2 = HTML(value="<h1><b>Product and Failure Distribution</b></h1>")
	TEMPLATE = "plotly_white"

	def __init__(self, job_input):
		self.auto_df = ProcessAutomation(job_input).df

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

	def plot_tn_graph(self):
		"""
        Plots the technology node distribution.

        Returns:
        --------
        tn_freq_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the technology node distribution plot.
        """
		tech_node_freq = self.auto_df['technology node'].value_counts()

		layout = go.Layout(title='Frequency of Technology Nodes', template='plotly_white')
		tn_freq_graph = go.FigureWidget(layout=layout)
		tn_freq_graph.add_bar(y=tech_node_freq.values, x=tech_node_freq.index)

		return tn_freq_graph

	def plot_prod_graph(self, tech_node):
		"""
        Plots the product distribution for the given technology node.

        Parameters:
        -----------
        tech_node : str
            The technology node for which to plot the product distribution.

        Returns:
        --------
        prod_freq_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the product distribution plot.
        """
		prod_name_freq = self.auto_df[self.auto_df['technology node']==tech_node]['product name'].value_counts()

		layout = go.Layout(title=f'Frequency of Product Names for {tech_node}', template='plotly_white')
		prod_freq_graph = go.FigureWidget(layout=layout)
		prod_freq_graph.add_bar(y=prod_name_freq.values, x=prod_name_freq.index)
    
		return prod_freq_graph

	def plot_fail_graph(self, tech_node, prod_name):
		"""
        Plots the failure distribution for the given technology node and product.

        Parameters:
        -----------
        tech_node : str
            The technology node for which to plot the failure distribution.
        prod_name : str
            The product for which to plot the failure distribution.

        Returns:
        --------
        fail_freq_graph : plotly.graph_objs._figurewidget.FigureWidget
            A plotly FigureWidget object containing the failure distribution plot.
        """
		failure_freq = self.auto_df[(self.auto_df['technology node']==tech_node) & (self.auto_df['product name']==prod_name)]['failure'].value_counts()

		layout = go.Layout(title=f'Frequency of Failure for {prod_name} in {tech_node}', template='plotly_white')
		fail_freq_graph = go.FigureWidget(layout=layout)
		fail_freq_graph.add_bar(y=failure_freq.values, x=failure_freq.index)

		return fail_freq_graph

	def update_prod_name_dropdown(self, change):
		"""
        Updates the product dropdown list based on the selected technology node.

        Parameters:
        -----------
        change : dict
            A dictionary containing information about the change event that triggered the method.

        Returns:
        --------
        None.
        """
		tech_node_value = self.tech_node_dropdown.value
		new_prod_name_options = list(self.auto_df[self.auto_df['technology node']==tech_node_value]['product name'].unique())
		self.prod_name_dropdown.options = new_prod_name_options
		self.prod_name_dropdown.value = new_prod_name_options[0]

	def refresh_prod_graph(self, evt):
		"""
        Refreshes the product distribution plot based on the selected technology node.

        Parameters:
        -----------
        b : bool
            A boolean value indicating whether the button was clicked or not.

        Returns:
        --------
        None.
        """
		prod_freq_graph_new = self.plot_prod_graph(self.tech_node_dropdown.value)
		self.prod_freq_graph.data[0]['x'] = prod_freq_graph_new.data[0]['x']
		self.prod_freq_graph.data[0]['y'] = prod_freq_graph_new.data[0]['y']
		self.prod_freq_graph.update_layout(title_text=f'Frequency of Product Names for {self.tech_node_dropdown.value}')

	def refresh_fail_graph(self, evt):
		"""
        Refreshes the failure distribution plot based on the selected technology node and product.

        Parameters:
        -----------
        b : bool
            A boolean value indicating whether the button was clicked or not.

        Returns:
        --------
        None.
        """
		fail_freq_graph_new = self.plot_fail_graph(self.tech_node_dropdown.value, self.prod_name_dropdown.value)
		self.fail_freq_graph.data[0]['x'] = fail_freq_graph_new.data[0]['x']
		self.fail_freq_graph.data[0]['y'] = fail_freq_graph_new.data[0]['y']
		self.fail_freq_graph.update_layout(title_text=f'Frequency of Failure for {self.prod_name_dropdown.value} in {self.tech_node_dropdown.value}')