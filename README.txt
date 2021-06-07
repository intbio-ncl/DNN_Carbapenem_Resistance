There are four files specified in this repository:
"standard_transformer" - A DNN architecture, whose source code is primarily constituted by this tutorial: https://www.tensorflow.org/text/tutorials/transformer 
                         In addition, this script contains an input pipeline, and evaluation functions, for protein primary structures.
"create_datasets" - A script containing functions that can be used to construct datasets suitable for the "standard_transformer" script.
"Write_Results" - A script used by the "standard_transformer" script to write results to a local directory
"logo_maker" - A script used by the "standard_transformer" script to create logos displaying the confidence of predictions for individual amino acid residues

Dependencies:
-TensorFlow 2
-Pandas dataframes
-numpy 
-matplotlib
-LogoMaker

######### HOW TO CREATE DATASETS FOR THE NEURAL NETWORK (create_datasets.py)  ############
-A local directory must be specified in line 6, before any function can be run in this script. This is the directory that datasets will be saved to.

-The "construct_ancestral_sequences" function takes an argument of a local directory for a ancestral reconstruction STATE file, and creates a CSV
file that specifies the sequences (DNA or protein) that the STATE file details as continuous sequences.

-The "construct_source_dataset" functions takes three arguments, the local directories of: The ancestral sequence CSV file created by the function previously described, the 
CONTREE file, and the sequence alignment file. This functions creates a CSV file that specifies features (input sequences) and labels (desired output sequences)
in a format that can be processed by the "standard_transformer" script. 

######### HOW TO TRAIN THE NEURAL NETWORK AND MAKE PREDICTIONS (standard_transformer.py) ###########
*Before the transformer is run, a local directory for results to be saved to, must be specified in line 9 of the script "Write_Results". This is the only
adjustment that needs to be made to the "Write_Results" script. All other comments in this section describe adjustments to "standard_transformer.py".

-The directory for the folder containing the prepared datasets, should be specified in line 651, for variable "dataset_folder_directory"
-The name of the dataset file (EXCLUDING the .csv file extension" should be specified in line 652, for variable "dataset_identity"
-The ancestor sequence that you want to predict a novel descendant for, should be specified in line 663, for variable "ancestor_sequence". This ancestor sequence
should *****NOT***** occur in the dataset that was specified for "dataset_identity". 
-Remember to have fun






