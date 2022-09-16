# Multi-digit Recognition Based on Convolutional Neural Network with Adversarial Training

This is a project based on the paper "Multi-digit number recognition from street view imagery using deep convolutional neural networks". https://arxiv.org/abs/1312.6082 

The datasets are available at http://ufldl.stanford.edu/housenumbers/. You can also simply run the ipynb, in which the datasets will be downloaded using functions in get_svhn.py. 

Firstly, we creat a 7-layer (5 convolutional, 2 fully connected) network. The files are: dense.py, analysis.ipynb, recognition_analysis_nonextra.ipynb and recognition_pipeline.ipynb 

Then, we try adversarial attack with less layers to examine the robustness of the network. The files are : main.py and moderler.py

For details, please refer to the file structure.

File structure

    --log
      --cnn_structure
        --events.out.tfevents.1576180494.instance-gpu2 (tensorboard events for showing the structure of the cnn model)
        
    --models
        --dense_extra (model trained with extra data (please refer to the report for details))

    --utils
        --cnn
            --dense.py (the model structure used for dense_extra)
        --adversarial_modeler.py (adversarial training model)
        --adversarial_tools.py (tools used for adversarial training)
        --data_extractor.py (extract trainable data from original format)
        --get_svhn.py (download and untar data)

    --adversarial_main.ipynb (model with adversarial attack. Training results are included)
    
    --analysis.ipynb (analysis of the results using dense_extra)

    --recognition_analysis_nonextra.ipynb (training process with analysis using only train data (please refer to the report for details of the results using different parameters))
    
    --recognition_pipeline.ipynb (training process for dense_extra)

To generate the model in models folder, you can run recoginition_pipeline.ipynb. To make analysis of the model, you can run analysis.ipynb. To train models and analyze them at we did in the report, you can run recognition_analysis_nonextra.ipynb. To test for adversarial learning, you can run adversarial_main.ipynb. 




