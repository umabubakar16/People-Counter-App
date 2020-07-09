# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...
The process behind converting custom layers involves two necessary custom layer extensions whcih are:- Custom Layer Extractor and Custom Layer Operation
Custom Layer Extractor: is responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial
Custom Layer Operation: is responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters


Some of the potential reasons for handling custom layers are...
1. Running some experimental layer on top of an existing supported layer.
2. allow model optimizer to convert specific model to Intermediate Representation.
3. The Layers you're trying to run uses unsupported input/ output shapes or formats.
4. You're trying to run a framework out of the support frameworks like Tensorflow, ONNX, Caffe etc.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...
Pre conversion model of the faster_rcnn_inception_v2_coco_2018_01_28 size i.e fozen_inference_graph.pb is 54.5 and  post conversion of the model i.e fozen_inference_graph.bin + fozen_inference_graph.xml is 50.9

The inference time of the model pre- and post-conversion was...
pre-conversion model Inference time: 
Average inference time: 1426 ms
Min inference time:  732 ms
Max inference time: 2120 ms 

Post-conversion model Inference time:
Average inference time: 700 ms 
Min inference time: 450 ms 
Max inference time: 950 ms

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Some of the potential use cases of the people counter app are

   1. Vehicle Detection, Tracking, and Speed Estimation
   2. Object detection and image classification
   3. COVID-19 social distancing detector
   4. Intrusion detection

Each of these use cases would be useful because...
 They are all about detecting and/or counting object in a video stream.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

    i. Lighting:- the light will be important in order to obtain a good result. Lighting is most assential factor which affects to result of model. We need input image with lighting because model can't predict so accurately if input image is dark. So monitored place must have lights.
    ii. Model accuracy:- Deployed edge model must have high accuracy because deployed edge model works in real time if we have deployed low accuracy model then it would give faulty results which is no good for end users.
    iii. Camera focal length:- High focal length gives you focus on specific object and narrow angle image while Low focal length gives you the wider angle. Now It's totally depend upon end user's reuqirements that which type of camera is required. If end users want to monitor wider place than high focal length camera is better but model can extract less information about object's in picture so it can lower the accuracy. In compare if end users want to monitor very narrow place then they can use low focal length camera.
    iv. Image size:- Image size totally depend upon resolution of image. If image resolution is better then size will be larger. Model can gives better output or result if image resolution is better but for higher resolution image model can take more time to gives output than less resolution image and also take more memory. If end users have more memory and also can manage with some delay for accurate result then higher resoltuion means larger image can be use.

## Model Used
The Model used in this project is faster_rcnn_inception_v2_coco which is fast in detecting people with less errors. Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.

## Enter the following command to download the model from Tensorflow Object Detection Model Zoo:

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

## Extract the tar.gz file by using the following command:

tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

## Change the directory to the extracted folder of the downloaded model using:

cd faster_rcnn_inception_v2_coco_2018_01_28

## Enter the following command to convert the TensorFlow model to Intermediate Representation (IR) or OpenVINO IR format:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

## To run the project from /home/workspace#, enter the following command:

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
