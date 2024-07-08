# scCaT
An explainable capsulating architecture for sepsis diagnosis transferring from single-cell RNA sequencing
![Image text](https://github.com/DM0815/scCaT/blob/master/Framework.jpg)

## Prerequisite
* python 3.7.12
* Tensorflow 2.4.1
* cudatoolkit 11.8
* R 4.3 (Only the drawing of AUPRC required)
* R package (precrec 0.14.4, reticulate 1.34.0)
  
## Getting started
1.Use Anaconda to create a Python virtual environment. Here, we will create a Python 3.7 environment named `scCaT`:

```cmd
conda create -n scCaT python=3.7.12
```

2.You can check whether the virtual environment was successfully created with the following command:

```cmd
conda env list
```

3.Activate your virtual environment:

```cmd
conda activate scCaT
```


4.Install tensorflow-gpu==2.4.1:

```
python -m pip install tensorflow-gpu==2.4.1
```

5.Add the current environment to the Jupyter Notebook kernel. Note that you should be in the "base" environment when running the following command:

```
python -m ipykernel install --user --name=scCaT --display-name scCaT
```

6.Use jupyter notebook and run './code/trainCaT.ipynb' to review the code. Other ipynb files can be found by name to understand the corresponding experiments


## Data PreProcessing 
The data preprocessing procedures were under 'data preprocessing' folder.

`./data/dataPreprocessing/readdata.py` contains related codes for data preprocessing stage.   
`./data/dataPreprocessing/changefdrlimit.py` changes the Fdr Threshold during experiment.   
`./data/dataPreprocessing/deleteSparseData.py` changes the Sparse Threshold during experiment.   
`./data/dataPreprocessing/read_specgenes.py` selects specific genes for comparison with other biomarkers.

## Building scCaT
scCaT was built based on capsule network and transformer. It was trained on single-cell RNA-seq data and then transferred to bulk RNA data for clinical practice. The details of building and training scCaT can be found below.

`./code/IntersectSC&Bulk.ipynb` extracts the common genes included in the scRNA-seq data and bulk RNA data.   
`./code/trainCaT.ipynb` builds and trains the model on scRNA-seq.   
`./code/CompBiomarker_onSC.ipynb` compares scCaT to the existing biomarkers and traditional machine learning models.   
`./code/TransferToBulk.ipynb` transferrs scCaT to bulk RNA data and evaluates its performance on validation cohorts.   
`./code/Rotation_testing.ipynb` performs transferring on one cohorts and testing on the other cohorts.   
`./code/visualization.ipynb` includes the visualization of the primary capsules, the capsule outputs, and each of the capsule dimensions.  

## Data and Results contained in the folders
As the size of models and figures are large,  we did not upload in this repo. If you are interested in our method, please download from google drive XXX. 
All the data can be access from the accession number stated in our paper.

`biomarkers` contains the predicted results of the existing biomarkers and traditional machine learning models on scRNA-seq data.   
`./data/dataBulk` contains the sepsis cohorts of microarray and bulk RNA-seq used in this study for the evaluation. The data can be downloaded from Gene Expression Omnibus (GEO) database.   
`./data/dataSC` contains the raw data and the processed data of the single-cell RNA-seq data of sepsis.
`modelsave` contains the model called trained on single-cell RNA-seq data and the model fine-tuned on bulk RNA data.


    
This framework can be generalized to rare disease diagnosis and phenotype detection that has only a few samples available. The details of the study can be found in our paper.
