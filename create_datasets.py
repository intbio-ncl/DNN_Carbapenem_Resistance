# Script for creation of datasets for prediction of novel VIM (carpapenem resistance) gene variants
import os
import pandas as pd

# Directory to which all datasets are to be saved e.g. "D:\work\dataset_folder
dataset_directory = ""


# Constructs ancestral sequence primary structures from source STATE file
def construct_ancestral_sequences(source_state_file):
    dataframe = pd.DataFrame(columns=("node_identity", "ancestral_sequence"))
    text = (open(source_state_file, "r")).readlines()
    df_row_counter = -1
    for line in text:
        # Skip notes at top of file
        if line.find("#") > -1:
            continue
        line_data = line.split()

        # Skip headers
        if len(line_data[0]) == 4:
            continue

        node_identity = str(line_data[0])
        residue_identity = str(line_data[2])

        # Switches to new row in dataframe when processing sequence for a new node
        new_node = True
        for row in dataframe.itertuples():
            if node_identity == row[1]:
                new_node = False
        if new_node:
            df_row_counter += 1
            dataframe.loc[df_row_counter, "node_identity"] = node_identity
            dataframe.loc[df_row_counter, "ancestral_sequence"] = residue_identity
        if not new_node:
            dataframe.loc[df_row_counter, "ancestral_sequence"] += residue_identity

    dataframe_name = (os.path.split(source_state_file)[-1]).split(".")[0] + "_ancestor"

    dataframe.to_csv(dataset_directory + "\\" + dataframe_name + ".csv")


# Constructs source dataset of features and labels from contree and ancestral sequence source files
def construct_source_dataset(ancestral_sequence_dataset, contree_file, alignment_file):
    contree_file_text = open(contree_file, "r").read()
    ancestral_sequence_dataframe = pd.read_csv(ancestral_sequence_dataset, index_col=0)
    final_dataset_dataframe = pd.DataFrame(columns=("node_identity", "descendent_id", "ancestral_sequence",
                                                    "descendant_sequence"))

    # Retrieving node identities of tree sequences
    counter = 0
    for sample in open(alignment_file, "r").readlines()[1:]:
        counter += 1
        sample = sample.split()

        if len(sample) == 0:
            break

        # Retrieving DNA sequence for tree sample
        tree_sequences = open(alignment_file, "r").read()
        tree_sequences = tree_sequences.splitlines()
        number_of_samples = int(tree_sequences[0].split()[0])
        sequence_length = int(tree_sequences[0].split()[1])
        sample_number = counter
        sequence = tree_sequences[sample_number].split()[1]
        alignment = sample_number
        while len(sequence) < sequence_length:
            alignment += 1 + number_of_samples
            sequence += tree_sequences[alignment]

        tree_sequence_identity = sample[0].replace("|", "_")
        contree_pos = contree_file_text.find(tree_sequence_identity)
        contree_file_text_slice = contree_file_text[contree_pos:]
        close_bracket_pos = contree_file_text_slice.find(")")
        open_bracket_pos = contree_file_text_slice.find("(")
        if open_bracket_pos < close_bracket_pos:
            print("node could not be identified for", tree_sequence_identity)
            continue
        else:
            node_identity = (contree_file_text_slice[close_bracket_pos + 1:close_bracket_pos + 4])
            node_identity = node_identity.replace(":0", "").replace(":", "")

        # Retrieving ancestral sequence for node
        ancestral_sequence = False
        for ancestor_sample in ancestral_sequence_dataframe.itertuples():
            if ancestor_sample[1].replace("Node", "") == str(node_identity):
                ancestral_sequence = ancestor_sample[2]
                break
        if not ancestral_sequence:
            print("ancestral sequence could not be found for node", node_identity)
        else:
            final_dataset_dataframe.loc[
                counter] = node_identity, tree_sequence_identity, ancestral_sequence, sequence.upper()

    dataset_name = (os.path.split(ancestral_sequence_dataset)[-1]).split(".")[0].split("_")[0] + "_source_dataset"
    final_dataset_dataframe.reset_index(drop=True, inplace=True)
    final_dataset_dataframe.to_csv(dataset_directory + "\\" + dataset_name + ".csv")


# Combines two datasets together, e.g. VIM73 and VIM43 feature-label datasets
def combine_datasets(dataset_1, dataset_2):
    dataframe_1 = pd.read_csv(dataset_1, index_col=0)
    dataframe_2 = pd.read_csv(dataset_2, index_col=0)
    dataframe_1 = dataframe_1.append(dataframe_2, ignore_index=True)

    dataframe_1_name = (os.path.split(dataset_1)[-1]).split(".")[0]
    dataframe_2_name = (os.path.split(dataset_2)[-1]).split(".")[0]
    dataframe_1.to_csv(dataset_directory + "\\" + dataframe_1_name + "_" + dataframe_2_name + "_combined.csv")
