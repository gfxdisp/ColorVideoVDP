# This is an example how to write code for genarating HTML reports
# You can run this code from the root directory of this repository as:
#
# PYTHONPATH=./ python gfxdisp/html_report/html_report_example.py
#
from matplotlib import colors
from matplotlib.markers import MarkerStyle
from gfxdisp.html_report import html_report
import numpy as np
import math
import os.path

import matplotlib.pyplot as plt

header_file = os.path.join(os.path.dirname(__file__), "header_example.html")

with html_report( "report_example/index.html", header_file=header_file ) as hr:

    hr.println( "<h1>Just a header text</h1>" )

    hr.beg_table( id_tag='datasets', class_tag='stripe' )
    hr.beg_table_row(ishead=True)
    hr.insert_table_cells( "Text", "Image", "Plot" )
    hr.end_table_row()

    for it in range(10):

        hr.beg_table_row()

        hr.beg_table_cell()
        hr.println( "Item {:02d}".format(it) )
        hr.end_table_cell()

        hr.beg_table_cell()
        image = np.random.rand( 768, 1024, 3 )*(it/10/2)+0.5
        hr.insert_image( image, name="img_{:02d}".format(it), extension="png" )
        hr.end_table_cell()

        hr.beg_table_cell()
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.grid()
        x = np.linspace(0, 2*math.pi, 1024)
        ax.plot( x, np.sin(x*(it+1)) )
        ax.set_xlabel( 'Time')
        ax.set_ylabel( 'Amplitude')            
        hr.insert_figure( "fig_{:02d}".format(it) )
        plt.close()
        hr.end_table_cell()

        hr.end_table_row()

    hr.end_table()
