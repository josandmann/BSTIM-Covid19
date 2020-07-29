from plot_curves_window import curves as curves_window
from plot_curves_window_trend import curves as curves_window_trend
from plot_map_window import curves as germany_map
from plot_interaction_kernel_window import interaction_kernel
from shared_utils import make_county_dict
import os

start = int(os.environ["DATE_ID"]) 
county_dict = make_county_dict()

for c in county_dict.keys():
    curves_window(start, c, n_weeks=3, model_i=35, save_plot=True)
    curves_window_trend(start, c, save_plot=True)
    
germany_map(start, save_plot=True)
interaction_kernel(start, save_plot=True)
