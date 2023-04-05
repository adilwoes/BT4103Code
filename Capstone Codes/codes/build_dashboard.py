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

from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.enum.text import PP_ALIGN

from codes.Tool_Util_BU_Loading import ToolUtilBULoading
from codes.Anomaly_Detection import Anomaly_Detection
from codes.TAT_QBreakdown_dashboard import TatQBreakdown

files = {
    "Data/M4_Utilization_Sheet.txt": "M4",
    "Data/M5_Utilization_Sheet.txt": "M5",
    "Data/CP_Utilization_Sheet.txt": "CP"
}

#file_names = [f for f in os.listdir(directory) if f.endswith('.xlsm')]
file_names = ['Data/Singapore_Device_Priority_2022 - WW09']
analyse_year = re.findall('\d+', file_names[0])[0]

tool_excel_file = 'Data/Job Input Form_cleaned.xlsm'
anomaly_detection_file_path = 'Data/powerbi.xlsx'
tat_qbreakdown_file_path = f'Data/Singapore_Device_Priority_{analyse_year} - Calculated.xlsx'
tat_qbreakdown_file_path_cancelled = f'Data/Singapore_Device_Priority_{analyse_year} - Cancelled.xlsx'
ww = 'Data/Work Week Calendar.xlsx'

class BuildDashboard:
    tab_titles = ['KPI', 'KPI (others)', 'BU Loading', 'Tool Utilization', \
                  'Technology Nodes', 'Technology Anomaly Detection Rate']
    def __init__(self):
        self.tat_qbreakdown = TatQBreakdown(tat_qbreakdown_file_path, tat_qbreakdown_file_path_cancelled, ww)
        self.anomaly_detection = Anomaly_Detection(anomaly_detection_file_path)
        self.tool_bu_dashboard = ToolUtilBULoading(files, tool_excel_file)
        self.export_to_ppt = Button(description='Export KPI Graphs to PPT', button_style='warning', \
                                    layout = Layout(width='auto'))
        self.export_to_ppt1 = Button(description='Export All Graphs to PPT', button_style='warning', \
                                     layout = Layout(width='auto'))
        self.export_to_ppt.on_click(self.create_ppt)
        self.export_to_ppt1.on_click(self.create_ppt_all)
        
        self.dashboard = Tab([self.tat_qbreakdown.kpi_tab, self.tat_qbreakdown.kpi_others_tab, \
               self.tool_bu_dashboard.tab2, self.tool_bu_dashboard.tab3, self.tool_bu_dashboard.tab, self.anomaly_detection.tab])
        [self.dashboard.set_title(i, title) for i, title in enumerate(self.tab_titles)]
        
        self.export = HBox([self.export_to_ppt, self.export_to_ppt1])
        self.app = VBox([self.export, self.dashboard])

    def save_png_tab1(self):
        turnaround_graph = self.tat_qbreakdown.kpi_tab.children[4]
        priority_graph = self.tat_qbreakdown.kpi_tab.children[7]
        csc_graph = self.tat_qbreakdown.kpi_tab.children[9]
        bu_graph = self.tool_bu_dashboard.tab2.children[2]
        tool_util = self.tool_bu_dashboard.tab3.children[2]
        
        type_graph = self.tat_qbreakdown.kpi_others_tab.children[4]
        type_month_graph = self.tat_qbreakdown.kpi_others_tab.children[6]
        prod_graph = self.tat_qbreakdown.kpi_others_tab.children[8]
        
        tech_node = self.tool_bu_dashboard.tab.children[1].children[0]
        prod_dist = self.tool_bu_dashboard.tab.children[3].children[0].children[1]
        failure_dist = self.tool_bu_dashboard.tab.children[3].children[1].children[1]
        
        #os.makedirs(r'Output')  
        tool_util.write_image('Output/Tool Utilization.png')
        bu_graph.write_image('Output/BU Loading.png')
        turnaround_graph.write_image('Output/TAT.png')
        priority_graph.write_image('Output/High Priority Queue Time Breakdown.png')
        csc_graph.write_image('Output/Overall Analysis Status.png')
        
        type_graph.write_image('Output/Type Breakdown.png')
        type_month_graph.write_image('Output/Type Breakdown by Month.png')
        prod_graph.write_image('Output/Product Breakdown.png')
        
        tech_node.write_image('Output/Technology Node Distribution.png')
        prod_dist.write_image('Output/Technology Product Distribution.png')
        failure_dist.write_image('Output/Technology Failure Distribution.png')


    def create_ppt(self, evt):
        self.save_png_tab1()

        X = Presentation('Powerpoint Output/SGP FI_KPI_Template.pptx')
        #Layout = X.slide_layouts[5]
        first_slide = X.slides[0]
        #first_slide.shapes.title.text = ''
        first_slide.shapes.add_picture('Output/TAT.png', Inches(0.2), Inches(1.5), height = Inches(2.7))
        first_slide.shapes.add_picture('Output/High Priority Queue Time Breakdown.png', Inches(4.6), Inches(1.5), height = Inches(2.7))
        first_slide.shapes.add_picture('Output/Overall Analysis Status.png', Inches(9.2), Inches(1.5), height = Inches(2.7))
        first_slide.shapes.add_picture('Output/BU Loading.png', Inches(3), Inches(4.5), height = Inches(2.7))
        first_slide.shapes.add_picture('Output/Tool Utilization.png', Inches(7.6), Inches(4.5), height = Inches(2.7))

#         text_frame = first_slide.shapes[0].text_frame

#         p = text_frame.paragraphs[0]
#         p.alignment = PP_ALIGN.LEFT
#         run = p.add_run()
#         run.text = 'SGP FI'

#         font = run.font
#         font.name = 'Calibri'
#         font.size = Pt(30)
#         font.bold = True

        X.save('Powerpoint Output/SGP FI_KPI.pptx')  
        
    def create_ppt_all(self, evt):
        self.save_png_tab1()

        X = Presentation('Powerpoint Output/SGP FI_KPI_Template.pptx')
        #Layout = X.slide_layouts[5]
        first_slide = X.slides[0]
        #first_slide.shapes.title.text = ''
        first_slide.shapes.add_picture('Output/TAT.png', Inches(0.2), Inches(1.5), height = Inches(2.2))
        first_slide.shapes.add_picture('Output/High Priority Queue Time Breakdown.png', Inches(4.6), Inches(1.5), height = Inches(2.2))
        first_slide.shapes.add_picture('Output/Overall Analysis Status.png', Inches(9.2), Inches(1.5), height = Inches(2.2))
        first_slide.shapes.add_picture('Output/BU Loading.png', Inches(3), Inches(4), height = Inches(2.2))
        first_slide.shapes.add_picture('Output/Tool Utilization.png', Inches(7.6), Inches(4), height = Inches(2.2))
        
        Layout = X.slide_layouts[5]
        second_slide = X.slides.add_slide(Layout)
        second_slide.shapes.title.text = ''
        second_slide.shapes.add_picture('Output/Type Breakdown.png', Inches(0.2), Inches(1.5), height = Inches(2.7))
        second_slide.shapes.add_picture('Output/Type Breakdown by Month.png', Inches(4.6), Inches(1.5), height = Inches(2.7))
        second_slide.shapes.add_picture('Output/Product Breakdown.png', Inches(9.2), Inches(1.5), height = Inches(2.7))
        second_slide.shapes.add_picture('Output/Technology Node Distribution.png', Inches(0.2), Inches(4.5), height = Inches(2.7))
        second_slide.shapes.add_picture('Output/Technology Product Distribution.png', Inches(4.6), Inches(4.5), \
                                        height = Inches(2.7))
        second_slide.shapes.add_picture('Output/Technology Product Distribution.png', Inches(9.2), Inches(4.5), \
                                        height = Inches(2.7))

#         text_frame = first_slide.shapes[0].text_frame

#         p = text_frame.paragraphs[0]
#         p.alignment = PP_ALIGN.LEFT
#         run = p.add_run()
#         run.text = 'SGP FI'

#         font = run.font
#         font.name = 'Calibri'
#         font.size = Pt(30)
#         font.bold = True

        X.save('Powerpoint Output/SGP FI_KPI (ALL).pptx')  
   
     