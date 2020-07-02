#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore



class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    
    def load_model(self, model, CPU_EXTENSION, DEVICE, console_output= False):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()
        
         # Add a CPU extension, if applicable
        if CPU_EXTENSION and "CPU" in DEVICE:
            self.plugin.add_extension(CPU_EXTENSION, DEVICE)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        supported_layers = self.plugin.query_network(network=self.network, device_name=DEVICE)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Following layers are not supported: ")
            for i in range(len(unsupported_layers)):
                print (i+1, ". ",unsupported_layers[i])
            exit(1)
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, DEVICE)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shapes = {}
        for inp in self.network.inputs:
            input_shapes[inp] = (self.network.inputs[inp].shape)
        return input_shapes
    

    def exec_net(self, net_input, request_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.exec_network.start_async(
                request_id, 
                inputs=net_input)
        return
    
    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[0].wait(-1)
        return status
    

    def extract_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
         
    
    