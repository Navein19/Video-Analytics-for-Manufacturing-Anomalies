# Fault-Detection-in-FBBC-Assembly
An object detection algorithm for the fault detection and monitoring system in FBBC manufacturing assembly



The basic requirements are:
Installing Anaconda, python >=3.5


***********************************************************************************************************

GPU USERS - Prefered requirements: RAM 8gb+ , GPU - NVIDIA (gtx 1050 or higher)
for gpu : additional requirements: Installing CUDA and CUDnn and setup the environment variables correctly.

***********************************************************************************************************

The required packages to download are:
1) Model Zoo 
2)Object Detetction Model folder
3) Label IMG Utility for creating custom dataset (https://github.com/tzutalin/labelImg#)
(Both 1,2 are uploaded onto the model already)

## - Inside Anaconda Prompt
# - commands under Anaconda Prompt


STEP 1: creating an environment:
# conda create -n <your env name>(say tf1.12) pip python=3.5

NOTE: It is better to use python 3.5 with tensorflow 1.12

STEP 2:Activate the environment

# conda activate tf1.12

--> Lets say your base is at C:/abc/desktop
Then download this model and extract onto a new folder (say tf1.12) 
Therefore, the working directory becomes : C:/abc/desktop/tf1.12

--> Inside the C:/abc/desktop/tf1.12 extract the above model.

STEP 3:(Under tf1.12 - your environment)
INSTALLING TENSORFLOW

# pip install --ignore-installed --upgrade tensorflow==1.12 (FOR CPU)
# pip install --ignore-installed --upgrade tensorflow-gpu==1.12(FOR GPU)

STEP 4: Installing required packages:

# conda install -c anaconda protobuf
# pip install pillow lxml cython pandas jupyter-notebook matplotlib opencv-python (pip install separately)


STEP 5: Configuring Pythonpath :
under (#)
# C:/abc/desktop/tf1.12/models/research : set PYTHONPATH = C:/abc/desktop/tf1.12/models
# C:/abc/desktop/tf1.12/models/research : set PYTHONPATH = C:/abc/desktop/tf1.12/models/research
# C:/abc/desktop/tf1.12/models/research : set PYTHONPATH = C:/abc/desktop/tf1.12/models/research/slim

STEP 6: Compliling protobufs:
under (#)
# C:/abc/desktop/tf1.12/models/research: protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection/protos/argmax_matcher.proto .\object_detection/protos/bipartite_matcher.proto .\object_detection/protos/box_coder.proto .\object_detection/protos/box_predictor.proto .\object_detection/protos/calibration.proto .\object_detection/protos/eval.proto .\object_detection/protos/faster_rcnn.proto .\object_detection/protos/faster_rcnn_box_coder.proto .\object_detection/protos/flexible_grid_anchor_generator.proto  .\object_detection/protos/graph_rewriter.proto .\object_detection/protos/grid_anchor_generator.proto .\object_detection/protos/hyperparams.proto .\object_detection/protos/image_resizer.proto .\object_detection/protos/input_reader.proto .\object_detection/protos/keypoint_box_coder.proto .\object_detection/protos/losses.proto .\object_detection/protos/matcher.proto .\object_detection/protos/mean_stddev_box_coder.proto .\object_detection/protos/model.proto .\object_detection/protos/multiscale_anchor_generator.proto .\object_detection/protos/optimizer.proto .\object_detection/protos/pipeline.proto .\object_detection/protos/post_processing.proto .\object_detection/protos/preprocessor.proto .\object_detection/protos/region_similarity_calculator.proto .\object_detection/protos/square_box_coder.proto .\object_detection/protos/ssd.proto .\object_detection/protos/ssd_anchor_generator.proto .\object_detection/protos/string_int_label_map.proto .\object_detection/protos/train.proto

and

now run:
# python setup.py build
# python setup.py install

(if error occurs try changing dir to researh/slim and then execute the above commands)

STEP 7: Creating Custom dataset using labelIMG utility:
* Download the given labelIMG utility from the given link above.
*open anaconda prompt and execute the labelIMg.py file


#follow https://pythonprogramming.net/custom-objects-tracking-tensorflow-object-detection-api-tutorial/ for creating custom dataset.

STEP 8: generating CSV file.
under C:/abc/desktop/tf1.12/models/research/object_detection:
# python xml_to_csv.py

* now your csv files will be in images/ folder

Step 9: Generating TF RECORD

go to generate_tfrecord.py:
and change the row_label to your required class label to detect.
and under the else condition return 0 in class_text_to_int(row_label) function.

for generating tfrecords: 
# 
# Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv  --output_path=/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --output_path=/test.record

STEP 10:
Go to C:/abc/desktop/tf1.12/models/research/object_detection/samples/configs 
* copy faster_rcnn_inception_v2_pets.config
and paste in C:/abc/desktop/tf1.12/models/research/object_detection/training folder

configure the training path in the config file accordingly.
and change the number of classes to your size of the test xml files and also set batch_size=1

STEP 11: TRAINING THE MODEL
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

STEP 12: EXPORT THE INFERENCE GRAPH
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_inception_v2_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856(use your value) \
    --output_directory (any name)_inference_graph
 
 STEP 13:
 Finally run. object_detection_video.py
 



