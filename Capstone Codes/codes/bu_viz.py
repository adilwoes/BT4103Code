from ipywidgets import Button, HBox, VBox, HTML, Dropdown
import ipywidgets as widgets
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output, display

from codes.process_automation import ProcessAutomation

class BUViz:
    """
    A class for visualizing BU Loading data.

    Parameters:
    -----------
    job_input: str
        The file path of the Job Input Form Excel file.

    TITLE: str
        A constant string representing the title of the BU loading stacked barchart visualization.
    TEMPLATE : str
        A constant string representing the plotly template to be used for all the visualizations.
    COLOR_CODES : list
        A list of color codes to be used for different bars in the visualizations.
    auto_df: pandas.DataFrame
        A DataFrame containing the processed and cleaned Job Input Form DataFrame. 
    output: widgets.Output
        An Output widget used to display the visualization.
    all_months: list
        A sorted list of all available year-months in auto_df.
    start_month_dropdown: widgets.Dropdown
        A dropdown widget used to select the start year_month for the visualization.
    end_month_dropdown: widgets.Dropdown
        A dropdown widget used to select the end year_month for the visualization.
    filtered_bu_df: pandas.DataFrame
        A DataFrame containing the filtered BU data for the selected year months.
    stacked_barchart: plotly.graph_objs._figurewidget.FigureWidget
        The barchart object for the BU Loading visualization.
    bu_graph: widgets.VBox
        The vertical box widget for displaying the barchart.
    month_selector: widgets.HBox
        An HBox widget containing the start_month_dropdown and end_month_dropdown widgets.
    tab: widgets.VBox
        A VBox widget containing the TITLE, month_selector, and stacked_barchart widgets.
    """
    TITLE = HTML(value="<h1><b>SGP BU Loading</b></h1>")
    TEMPLATE = "plotly_white"
    COLOR_CODES = ['#ff7f0e', '#1f77b4', '#FFBF5F', '#aec7e8', '#7f7f7f']

    def __init__(self, job_input):
            self.auto_df = ProcessAutomation(job_input).job_input_df

            self.output = widgets.Output()

            self.all_months = sorted(self.auto_df['Year Month'].unique())
            self.start_month_dropdown = Dropdown(
                options=self.all_months, 
                description='Start Month:', style={'description_width': 'initial'})
            self.end_month_dropdown = Dropdown(
                options=self.all_months, 
                value=self.all_months[-1], 
                description='End Month:', style={'description_width': 'initial'})

            self.start_month_dropdown.observe(self.on_month_filter_change, names='value')
            self.end_month_dropdown.observe(self.on_month_filter_change, names='value')

            self.filtered_bu_df = self.auto_df[(self.auto_df['Year Month'] >= self.start_month_dropdown.value) & (self.auto_df['Year Month'] <= self.end_month_dropdown.value)]
            self.stacked_barchart = self.plot_bu_barchart(self.filtered_bu_df)

            with self.output:
                display(self.stacked_barchart)

            self.bu_graph = VBox([self.output])
            self.month_selector = HBox([self.start_month_dropdown, self.end_month_dropdown])
            self.tab = VBox([self.TITLE, self.month_selector, self.stacked_barchart])

    def plot_bu_barchart(self, df):
        """
        Plots the BU loading stacked barchart.

        Parameters:
        -----------
        df: pandas.DataFrame
            A DataFrame containing the BU Loading data.

        Returns:
        --------
        FigureWidget
            A FigureWidget object containing the visualization of the BU Loading stacked barchart.
        """
        temp = df.groupby(['Year Month', 'BU'])['Job ID'].count().unstack().fillna(0).reset_index()

        fig = go.FigureWidget(data=[go.Bar(x=temp['Year Month'], y=temp[col], name=col, marker=dict(color=color)) for color, col in zip(self.COLOR_CODES, temp.columns[1:])])
        fig.update_layout(barmode='stack', xaxis_title='Year-Month', yaxis_title='Job Count', title="BU Loading", template=self.TEMPLATE)

        return fig

    def on_month_filter_change(self, change):
        """
        Updates the BU loading stacked barchart based on the selected months.

        Parameters:
        -----------
        change : dict
            A dictionary containing information about the change event that triggered the method.

        Returns:
        --------
        None.
        """
        with self.output:
            clear_output(wait=True)
            filtered_bu_df = self.auto_df.loc[(self.auto_df['Year Month'] >= self.start_month_dropdown.value) & (self.auto_df['Year Month'] <= self.end_month_dropdown.value)]
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
