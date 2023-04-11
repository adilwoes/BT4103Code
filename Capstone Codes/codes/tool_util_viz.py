from ipywidgets import HBox, VBox, HTML, Dropdown
import ipywidgets as widgets
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output, display

from codes.process_automation import ProcessToolUtilization

class ToolUtilViz:
    """
    A class that creates a visualization of the tool utilization rate 
    for a given set of tool files.

    Parameters:
    -----------
    tool_files : dict
        A dictionary mapping file paths to tool labels.

    Attributes:
    -----------
    TITLE: HTML
        A class constant representing the HTML title of the visualization.
    TEMPLATE: str
        A class constant representing the template used for the visualization.
    tool_df: pandas.DataFrame
        A DataFrame containing the processed tool utilization data.
    output: widgets.Output
        An Output widget used to display the visualization.
    all_weeks: list
        A sorted list of unique work weeks in the tool utilization data.
    start_week_dropdown: widgets.Dropdown
        A dropdown widget used to select the start week for the visualization.
    end_week_dropdown: widgets.Dropdown
        A dropdown widget used to select the end week for the visualization.
    filtered_tool_df: pandas.DataFrame
        A DataFrame containing the filtered tool utilization data for the selected work weeks.
    util_chart: plotly.graph_objs._figurewidget.FigureWidget
        A FigureWidget object containing the visualization of the tool utilization rate.
    util_graph: widgets.VBox
        A VBox widget containing the output2 widget and other widgets for the visualization.
    week_selector: widgets.HBox
        An HBox widget containing the start_week_dropdown and end_week_dropdown widgets.
    tab: widgets.VBox
        A VBox widget containing the TITLE, week_selector, and util_chart widgets.
    """

    TITLE = HTML(value="<h1><b>Tool Utilization Rate (Weekly)</b></h1>")
    TEMPLATE = "plotly_white"
    
    def __init__(self, tool_files):
        self.tool_df = ProcessToolUtilization(tool_files).final_df

        self.output = widgets.Output()

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

        with self.output:
            display(self.util_chart)

        self.util_graph = VBox([self.output])
        self.week_selector = HBox([self.start_week_dropdown, self.end_week_dropdown])
        self.tab = VBox([self.TITLE, self.week_selector, self.util_chart])
    
    def plot_weekly_util_rate(self, df):
        """
        Plots the weekly tool utilization rate.

        Parameters:
        -----------
        df: pandas.DataFrame
            A DataFrame containing the tool utilization data.

        Returns:
        --------
        FigureWidget
            A FigureWidget object containing the visualization of the tool utilization rate.
        """
        layout = go.Layout(title='Tool Utilization Rate', template='plotly_white', yaxis=dict(tickformat=".0%"))

        util_rate_graph = go.FigureWidget(layout=layout)

        for tool in df.tool.unique():
            temp_tool = df.loc[df.tool == tool]
            util_rate_graph.add_scatter(y=temp_tool.final_rate, x=temp_tool.work_week, name=tool)

        return util_rate_graph

    def on_week_filter_change(self, evt):
        """
        Updates the visualization when the week selection is changed.

        Parameters:
        -----------
        evt: dict
            A dictionary containing information about the event that triggered the update.
        """
        with self.output:
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