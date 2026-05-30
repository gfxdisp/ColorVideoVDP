import os.path
from PIL import Image
import numpy as np

from pathlib import Path
import shutil

import matplotlib

class html_report:

    def __init__(self, file_name, header_file=None):

        self.name_ind = 0  # to generate unique file names        
        self.path, file = os.path.split(file_name)

        self.thead_open = False
        self.tbody_open = False

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self.html_file_name = file_name
        self.fh = open(file_name, "w")

        self.println( "<html>")
        self.println( "<head>")
        self.insert_file(os.path.join(os.path.dirname(__file__), "html_preable.html"))
        if header_file:
            self.insert_file(header_file)
        self.println( "</head>")
        self.println( "<body>")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def print(self, str):
        self.fh.write( str )

    def println(self, str):
        self.fh.write( str + "\n" )

    def copy_file(self, file_name, dest_rel_path='./' ):
        path, file = os.path.split(file_name)
        source = Path(file_name)
        destination = Path(self.path) / dest_rel_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)

    def insert_file(self, fname):
        with open(fname) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.println(line)

    def __get_img_name( self, name=None, extension="png"):
        if not name:
            name = "img_{ind:04d}".format(ind=self.name_ind)
            self.name_ind += 1

        img_name = name + "." + extension
        full_name = os.path.join(self.path, img_name )
        return full_name, img_name


    def insert_figure(self, name=None):
        full_name, img_name = self.__get_img_name(name, ".png")
        matplotlib.pyplot.savefig(full_name)
        self.println( "<img src='{img}'/>".format( img=img_name ) )

    def __save_image( self, im, name=None, extension="png" ):
        full_name, img_name = self.__get_img_name(name, extension)
        im = Image.fromarray((im.clip(0,1)*255).astype(np.uint8))
        im.save(full_name)
        return img_name

    def insert_image(self, im, name=None, extension="png", mousedown_image=None):
        img_name = self.__save_image(im, name, extension)
        # full_name, img_name = self.__get_img_name(name, extension)
        # im = Image.fromarray((im.clip(0,1)*255).astype(np.uint8))
        # im.save(full_name)

        if not mousedown_image is None:
            id_name = os.path.splitext(img_name)[0]
            if type(mousedown_image) is str:
                md_image_name = mousedown_image
            else:
                md_name = name + "_md"
                md_img_name = self.__save_image(mousedown_image, md_name, extension )

            extra_tags='name="{id}" unselectable="on" onDragStart="return false;" onMouseDown=\'if(detectLeftButton(event)){{document.{id}.src="{md_image}";return true;}}\' onMouseUp=\'document.{id}.src="{image}";return true;\' '.format(id=id_name, md_image=md_img_name, image=img_name)

        else:
            extra_tags=''

        self.println( "<img {ext_tags}src='{img}'/>".format( ext_tags=extra_tags, img=img_name ) )

    def beg_table(self, id_tag=None, class_tag=None):
        self.print("<table")
        if id:
            self.print(" id={id}".format(id=id_tag))
        if class_tag:
            self.print(" class={}".format(class_tag))
        self.print(">")

    def end_table(self):
        if self.tbody_open:
            self.println("  </tbody>")
            self.tbody_open=False
        self.println("</table>")

    def beg_table_row(self, ishead=False):
        if ishead:
            if not self.thead_open:
                self.println("  <thead>")      
                self.thead_open=True
        else:
            if self.thead_open:
                self.println("  </thead>")      
                self.thead_open=False

            if not self.tbody_open:
                self.println("  <tbody>")      
                self.tbody_open=True

        self.println("    <tr>")      

    def end_table_row(self):
        self.println("    </tr>")      

    def beg_table_cell(self):
        if self.thead_open:
            self.println("      <th>")      
        else:
            self.println("      <td>")      

    def end_table_cell(self):
        if self.thead_open:
            self.println("      </th>")      
        else:
            self.println("      </td>")      

    def insert_table_cells(self,*args):
        if self.thead_open:
            tag="th"
        else:
            tag="td"
        for cell in args:
            self.println( "<{tag}>{cell}</{tag}>".format(tag=tag, cell=cell))

    def close( self ):
        self.println( "</body>")
        self.println( "</html>")
        self.fh.close()
        print( "HTML report generated in file://{}".format(os.path.abspath(self.html_file_name)) )

