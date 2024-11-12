# Copyright (C) 2024 Università degli Studi di Cagliari
# Licensed under BSD 3-Clause License (see LICENSE)
# Authors: Francesco Ratto, Federico Manca (<name>.<surname>@unica.it)

import os






# METHODS FOR GENERATING TCL FILE
#
# ci serve un metodo (generate_TCL_file) per mettere
# assieme Settings (get_TCL_settins) e fine file (get_TCL_ending).
# Al contrario del file XDF, non c'è bisogno di
# rappresentare le connessioni (get_XDF_connections).
class TCLWriter():

    def __init__(self, writer_node_list, path):

        self.node_list = writer_node_list
        self.path = path
        

        # recover data from reader node
        self.generate_TCL_file()
    # generates the TCL file
    def generate_TCL_file(self):

        # return the Project Settings of the TCL file
        prjset_TCL_file = self.get_TCL_settings()

        # return the ending of the TCL file
        ending_TCL_file = self.get_TCL_ending()

        # compose the TCL file
        TCL_file = prjset_TCL_file + ending_TCL_file

        # name of the directory where save the file
        dir_name = "/Cpp"
        
        # name of the directory where save the HLS
        dir_hls_name = "/HLS"

        # path where put create a directory which
        # will contain TCL file
        path = self.path + dir_name
        
        path_hls = self.path + dir_hls_name

        # create a directory for TCL file (if not existing)
        if os.path.exists(path) is not True:
            os.makedirs(path)
            
        #####
        if os.path.exists(path_hls):
            # Remove all files except fifo_gen.v
            for filename in os.listdir(path_hls):
                file_path = os.path.join(path_hls, filename)
                if filename != 'fifo_gen.v':
                    os.remove(file_path)
        else:
            # Create the folder if it doesn't exist
            os.makedirs(path_hls)
        

        # create TCL file
        with open(os.path.join(path, "TCL_file.tcl"), "w") as the_TCL_file:
            the_TCL_file.write(TCL_file)

    # generate TCL Project Settings
    def get_TCL_settings(self):

        prjset_TCL_file = ""
        
        template_init = \
            """
        package require fileutil
            """
        prjset_TCL_file += template_init
        
        back_template = \
            """
        cd ..
            """
        
        # for every writer node in the net
        for writer_node in self.node_list:

            id = writer_node.name

            template_init = \
            """
        cd {}

        open_project -reset {}

        set_top {}
    """
            # fill template with name of the project
            if("Conv" in id):
                id_conv = "pe_" + id
                prjset_TCL_file += template_init.format( id, id, id_conv)
            else:
                prjset_TCL_file += template_init.format( id, id, id)

            
            template_my_files = \
                    """ 
        add_files my_types_Input0.h
            """
            #prjset_TCL_file += template_my_files
            

            # template to be filled
            template = \
                """ 


        add_files {}.cpp

        add_files {}.h

        add_files layer_sizes_{}.h

        add_files my_types_{}.h

        add_files parameters_{}.h"""
            
            # fill template with name of the project
            if("Conv" in id):
                id_conv = "pe_" + id
                prjset_TCL_file += template.format(id_conv, id_conv, id, id, id)
                prjset_TCL_file += self.get_TCL_body(id,id)
            else:
                prjset_TCL_file += template.format(id, id, id, id, id, id, id, id)
                prjset_TCL_file += self.get_TCL_body(id,id)


            # particular care for Conv layer
            if (writer_node.operation == "MaxPool"):

                line_buff_template = \
                """

        open_project line_buffer_{}
            
        set_top line_buffer_{}
            
        add_files line_buffer_{}.cpp

        add_files line_buffer_{}.h"""
                
                # fill template with name of the project
                prjset_TCL_file += line_buff_template.format(id, id, id, id)
                prjset_TCL_file = prjset_TCL_file.replace("line_buffer_MaxPool","line_buffer_mp")
                
                id1 = "line_buffer_"
                id1 += id
                id1 = id1.replace("line_buffer_MaxPool","line_buffer_mp")
                
                
                prjset_TCL_file += self.get_TCL_body(id,id1)
                
                
                
                

            elif (writer_node.operation == "Conv"):
            
                line_buff_template = \
                """
                
        open_project line_buffer_{}
            
        set_top line_buffer_{}
            
        add_files line_buffer_{}.cpp

        add_files line_buffer_{}.h"""
                
                # fill template with name of the project
                prjset_TCL_file += line_buff_template.format(id, id, id, id)
                
                id1 = "line_buffer_"
                id1 += id
                
                prjset_TCL_file += self.get_TCL_body(id,id1)
                
                
            
                bias_template = \
                """
                
        open_project bias_{}

        set_top bias_{}

        add_files bias_{}.cpp

        add_files bias_{}.h
                
                """
                # fill template with name of the project
                prjset_TCL_file += bias_template.format(id, id, id, id)
                
                id1 = "bias_"
                id1 += id
                
                prjset_TCL_file += self.get_TCL_body(id,id1)
                
                
                
                
                weight_template = \
                """
            

        open_project weight_{}

        set_top weight_{}

        add_files weight_{}.cpp

        add_files weight_{}.h"""
                
                # fill template with name of the project
                prjset_TCL_file += weight_template.format(id, id, id, id)
                
                id1 = "weight_"
                id1 += id
                
                prjset_TCL_file += self.get_TCL_body(id,id1)
                
                
                
                
                

            prjset_TCL_file += back_template    

            

        # return TCL file
        return prjset_TCL_file

    # generate TCL ending
    def get_TCL_ending(self):
        # generate TCL ending string
        ending = \
            """
        exit"""

        # return string contaning TCL ending
        return ending
        
        
    # generate TCL body
    def get_TCL_body(self,id_1,id_2):
        is_board = False
        path = self.path.replace('\\','/')
        if not is_board:
            # generate TCL ending string
            solution_settings_template1 = \
                """ 
        open_solution -reset "solution1"

        set_part {"""

            board_part = \
        "xck26-sfvc784-2LV-c"

            solution_settings_template2 = \
                """}
        create_clock -period 10 -name default

        config_compile -enable_auto_rewind=0
        config_schedule -enable_dsp_full_reg=0


        csynth_design"""
        else:
            # generate TCL ending string
            solution_settings_template1 = \
                """ 
        open_solution -reset "solution1"

        set_part -board """
            board_part = \
        "xilinx.com:kv260:part0:1.1 "

            solution_settings_template2 = \
    """
        create_clock -period 10 -name default

        csynth_design"""

        

        solution_settings_template3 = \
                """           
                

        set sourceDir "{}/Cpp/{}/{}/solution1/syn/verilog"

        set sourceDirC "{}/Cpp/{}/"

        set destDir "{}/HLS"

        set destDirC "{}/Cpp"

        # Get a list of files in the source directory
        set fileList [glob -nocomplain -directory $sourceDir *]
        set fileListC [glob -nocomplain -directory $sourceDirC *.h *.cpp]

        foreach file $fileList"""

        Grafa1 = \
        """ {
        """

        solution_settings_template4 = \
                """
                # Create the destination path by joining the destination directory and the file name
                set destPath [file join $destDir [file tail $file]]

                file copy -force $file $destPath
                        """

        Grafa2 = \
                """
        }
                """
        Grafa3 = \
        """ 
        foreach file $fileListC {
        """

        solution_settings_template5 = \
                """
                # Create the destination path by joining the destination directory and the file name
                set destPath [file join $destDirC [file tail $file]]

                file copy -force $file $destPath
                        """

        Grafa4 = \
                """
        }
                """

        solution_settings_template6 = \
                """
        close_project
                """

        solution_settings = solution_settings_template1

        solution_settings += board_part

        solution_settings += solution_settings_template2

        solution_settings += solution_settings_template3.format(path, id_1, id_2, path, id_1, path, path)


        solution_settings += Grafa1

        solution_settings += solution_settings_template4

        solution_settings += Grafa2

        solution_settings += Grafa3

        solution_settings += solution_settings_template5
        
        solution_settings += Grafa4

        solution_settings += solution_settings_template6

        body = solution_settings                

        # return string contaning TCL ending
        return body

    # ----------------------------------------







