import ipywidgets as widgets
from IPython.display import display
import Data
import Prediction


class GUI:
    #declares all controls and views them
    def __init__(self,data_object):
        self.data_object = data_object
        self.prediction_label = widgets.Label(Value='')
        self.start_range = widgets.IntSlider(min=0,max=len(self.data_object.get_all_values()), description='Start:')
        self.value_viewer = widgets.Output()
        display(self.start_range, self.value_viewer)
        self.end_value = 0
        self.start_value = 0
        self.end_range = widgets.IntSlider(min=0,max=len(self.data_object.get_all_values()), description='End:')
        self.value_second_viewer = widgets.Output()
        display(self.end_range, self.prediction_label, self.value_second_viewer)
        self.value_second_viewer = widgets.Output()
        self.cut_button = widgets.Button(description="Cut and Predict")
        self.value_third_viewer = widgets.Output()
        display(self.cut_button, self.value_third_viewer)
        self.view_direction_prediction()
        self.assign_events()
    #calls the prediction method
    def view_direction_prediction(self):
        linearReg = Prediction.Prediction(self.data_object)
        self.prediction_label.value = "Direction: " + linearReg.predict_trend()
    #assign events to controls
    def assign_events(self):
        self.cut_button.on_click(self.on_button_clicked)
        self.start_range.observe(self.on_end_change, names='value')
        self.end_range.observe(self.on_start_change, names='value')
    #preforms slicing and prediction if triggered
    def on_button_clicked(self,b):
        with self.value_third_viewer:
            new_data = Data.Data()
            self.data_object.set_all_values(new_data.get_all_values()[self.end_value:self.start_value])
            self.view_direction_prediction()
    #sets the amount of start records
    def on_end_change(self,change):
        with self.value_viewer:
            self.end_value = change['new']
    #sets the amount of end records
    def on_start_change(self,change):        
        with self.value_viewer:
            self.start_value = change['new']