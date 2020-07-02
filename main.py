"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2
import os
import sys
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60



def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser



def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    video_file=args.input 
    
    # Flag for the input image
    single_img_flag = False
   
    ### TODO: Load the model through `infer_network` ###
    infer_network_vals = infer_network.load_model(args.model, args.cpu_extension, args.device)
    log.debug(infer_network_vals)
    network_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    # Check for live feed
    if video_file == 'CAM': 
        input_stream = 0
     # Check for input image
    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') or video_file.endswith('.png') or video_file.endswith('.jpeg'):   
        single_img_flag = True
        input_stream = video_file
    
    # Check for video file
    else:     
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file does not exist"
    
    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate: "+ video_file)
    except Exception as e:
        print("Something went wrong with your video file: ", e)

    ### TODO: Handle the input stream ###
    width = int(cap.get(3))
    height = int(cap.get(4))

    net_input_shape = network_shape['image_tensor']

    #iniatilize variables
    
    prev_duration = 0
    total_count = 0
    duration = 0
    request_id=0
    omitted_count = 0
    last_count = 0
    counter = 0
    previous_count = 0
    
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        inference_start_time = time.time()

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image,'image_info': image.shape[1:]}
        duration_report = None
        infer_network.exec_net(net_input, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            inference_time = time.time() - inference_start_time
            inference_time=round(inference_time,2)
            ### TODO: Get the results of the inference request ###
            net_output = infer_network.extract_output()

            ### TODO: Extract any desired stats from the results ###
         
            current_count = 0
            probs = net_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    current_count += 1
                    box = net_output[0, 0, i, 3:]
                    xmin,xmax = (int(box[0] * width), int(box[2] * width))
                    ymin,ymax = (int(box[1] * height), int(box[3] * height))
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count != counter:
                previous_count  = counter
                counter = current_count
                if duration >= 3:
                    prev_duration = duration
                    duration = 0
                else:
                    duration = prev_duration + duration
                    prev_duration = 0
            else:
                duration += 1
            ## When a person enters the frame
            if duration >= 3:
                    last_count = counter
                    client.publish('person',payload=json.dumps({'count': last_count}))
             ## When a person leaves the frame
            if duration == 3 and counter > previous_count:
                    total_count += counter - previous_count 
                    client.publish('person',payload=json.dumps({'total count': total_count}))
            #get total duration
            elif duration == 3 and counter < previous_count:
                    duration_report = int((prev_duration / 10.0) * 1000)
           
            ## to avoid counting person more than once
            ## person detected should be there for atleast 2sec
            if duration>=2:
                   total_count = total_count
            else:
                    # substract previous count from total_count
                    # and count it as omitted frame
                    total_count -=previous_count
                    omitted_count += 1
                    
             # Adding inference time to the frame  
            inf_time_message = "Inference time: {:.3f}ms".format(inference_time * 1000)
            cv2.putText(frame, inf_time_message, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        # Write an output image if `single_img_flag = true` 
        if single_img_flag:
            people_in_image = "Number of people in the image : "+int(current_count)
            cv2.putText(frame, people_in_image, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
            cv2.imwrite("output.jpg", frame)
        if key_pressed == 27:
            break
            
     # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()



def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()