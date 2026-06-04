import os
import pycvvdp
from tests.html_report import html_report
import pycvvdp.utils as utils
from datetime import datetime

import torch
import platform
import subprocess

import argparse

import re
import keyword

import matplotlib.pyplot as plt

from examples.ex_aliasing import aliasing_example
from examples.ex_simple_image import simple_image_example
from examples.ex_simple_video import simple_video_example
from examples.ex_display_brightness import display_brightness_example
from examples.ex_batch_of_images import batch_of_images_example
from examples.ex_batch_of_video import batch_of_video_example
from examples.ex_display_geometry import display_geometry_example

import pycvvdp.utils as utils

def current_date_yyyymmdd():
    return datetime.now().strftime("%Y_%m_%d")


def slugify_variable_name(name: str) -> str:
    # Lowercase and replace non-alphanumeric (except underscore) with underscores
    s = name.lower()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    # Collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    # Remove leading underscores/digits so it starts with a letter
    s = re.sub(r'^[^a-zA-Z]*', '', s)
    # If empty after cleanup, use a default
    if not s:
        s = "var"
    # Avoid Python keywords
    if keyword.iskeyword(s):
        s += "_"
    return s

def print_system_info( hr : html_report ):

    # Print PyTorch version
    hr.println( "<h2>System info</h2>" )
    hr.println( "<ul>" )
    hr.println(f"<li>PyTorch version: {torch.__version__}</li>")

    # Print CUDA version if present
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        hr.println(f"<li>CUDA version: {cuda_version}</li>")
    else:
        hr.println("<li>CUDA version: Not available</li>")

    # Get CPU name
    def get_cpu_name():
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":")[1].strip()
            except:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                    capture_output=True, text=True, check=True)
                return result.stdout.strip()
            except:
                pass
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(["wmic", "cpu", "get", "name"], 
                                    capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    return lines[-1].strip()
            except:
                pass
        return "Unknown CPU"

    hr.println(f"<li>CPU name: {get_cpu_name()}</li>")

    # Get GPU name
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        hr.println(f"<li>GPU name: {gpu_name}</li>")
    else:
        hr.println("<li>GPU name: No CUDA GPU available</li>")

    # Print OS version
    os_version = platform.platform()
    hr.println(f"<li>Operating system version: {os_version}</li>")

    hr.println( "</ul>" )


def test_all(device_name):

    device = utils.get_best_device(device_name)

    header_file = os.path.join(os.path.dirname(__file__), "header_example.html")

    parameters = utils.json2dict('pycvvdp/vvdp_data/cvvdp_parameters.json');
    version = parameters['version']

    metric_classes = [ pycvvdp.cvvdp, pycvvdp.cvvdp_ml_saliency, pycvvdp.cvvdp_ml_transformer ]

    metric_instances = [mc(device=device) for mc in metric_classes]
    metric_names = [met.short_name() for met in metric_instances]

    # TESTs = [ ('Display geometry', display_geometry_example) ]

    TESTs = [
        ('Simple image example', simple_image_example),
        ('Simple video example', simple_video_example),
        ('Aliasing', aliasing_example),
        ('Display brightness', display_brightness_example),
        ('Display geometry', display_geometry_example),
        ('Batch of images', batch_of_images_example),
        ('Batch of video', batch_of_video_example)
    ]
 
    with html_report( f"tests/test_report_{current_date_yyyymmdd()}-{version}-{str(device)}/index.html", header_file=header_file ) as hr:

        hr.copy_file( os.path.join(os.path.dirname(__file__), 'style.css'), './' )
        hr.println( "<h1>ColorVideoVDP test report</h1>" )

        hr.println( f"Testing started on {datetime.now().strftime('%A, %d %B %Y, %H:%M')}</br>" )

        hr.println( f"Running on <b>{str(device)}</b></br>" )

        hr.println( "<h2>Tested metrics</h2>" )
        hr.println( "<ul>" )
        for mi in metric_instances:
            hr.println( f"<li>{mi.short_name()} - {mi.get_info_string()}</li>" )
        hr.println( "</ul>" )

        print_system_info(hr)

        for tst_idx, tst in enumerate(TESTs):

            print( f"Running {tst[0]}" )
            test_id = slugify_variable_name(tst[0])
            hr.println( f'<h2 id={test_id}><a class="top_align_button" href="#{test_id}">&#8632;</a>{tst[0]}</h2>' )

            met_res = []
            for mc in metric_classes:
                testing_func = tst[1]
                met_res.append( testing_func(mc, device=device) )
            
            hr.beg_table( class_tag='stripe' )
            hr.beg_table_row(ishead=True)

            hr.insert_table_cells( *([ "Test" ] + metric_names) )    
            hr.end_table_row()

            for test_idx in range(len(met_res[0])):
                hr.beg_table_row()

                hr.beg_table_cell()
                hr.println( met_res[0][test_idx][0] )
                hr.end_table_cell()

                for mm in range(len(metric_classes)):
                    hr.beg_table_cell()
                    hr.println( f"{met_res[mm][test_idx][1]:.2f} {metric_instances[mm].quality_unit()} ({met_res[mm][test_idx][2]:.2f} secs)" )
                    hr.end_table_cell()

            hr.end_table()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ColorVideoVDP on a set of videos")
    parser.add_argument("--device", type=str,  default="auto", help="select which PyTorch device to use. Pick from ['cpu', 'mps', 'cuda', 'cuda:0', 'cuda:1', ...]")
    args = parser.parse_args()

    test_all(args.device)
