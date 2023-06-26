# CaT
An explainable capsulating architecture with transformer for sepsis detection transferring from single-cell RNA sequencing

## Data PreProcessing 
The data preprocessing procedures were under 'data preprocessing' folder.

`readdata.py` contains related codes for data preprocessing stage.   
`changefdrlimit.py` changes the Fdr Threshold during experiment.   
`deleteSparseData.py` changes the Sparse Threshold during experiment.   
`read_specgenes.py` selects specific genes for comparison with other biomarkers.

## Building CaT
CaT was built based on capsule network and transformer. It was trained on single-cell RNA-seq data and then transferred to bulk RNA data for clinical practice. The details of building and training CaT can be found below.

`IntersectSC&Bulk.ipynb` extracts the common genes included in the scRNA-seq data and bulk RNA data.   
`trainCaT.ipynb` builds the model and trains the model on scRNA-seq.   
`CompBiomarker_onSC.ipynb` compares CaT to the existing biomarkers and traditional machine learning models.   
`TransferToBulk.ipynb` transferrs CaT to bulk RNA data and evaluates its performance on validation cohorts.   
`Rotation_testing.ipynb` performs transferring on one cohorts and testing on the other cohorts.   
`visualization.ipynb` includes the visualization of the primary capsules, the capsule outputs, and each of the capsule dimensions.  

## Data and Results contained in the folders
As the size of models and figures are large,  we did not upload in this repo. If you are interested in our method, please feel free to contact me for the model at xbzheng@gbu.edu.cn.

`biomarkers` contains the results of the existing biomarkers and traditional machine learning models on scRNA-seq data.   
`dataBulk` contains the sepsis cohorts of microarray and bulk RNA-seq used in this study for the evaluation. The data can be downloaded from Gene Expression Omnibus (GEO) database.   
`dataSC` contains the raw data and the processed data of the single-cell RNA-seq data of sepsis.   
`model` contains the model called trained on single-cell RNA-seq data and the model fine-tuned on bulk RNA data.   
    
This framework can be generalized to rare disease diagnosis and phenotype detection that has only a few samples available. The details of the study can be found in our paper.
