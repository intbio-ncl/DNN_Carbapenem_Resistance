# Writes results, and saves images to a local directory

import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# The directory in which results are to be written
root_directory = ""  # The folder directory to which results are to be written, e.g. "D:\work\Results"

# Used to determine the most successful design across multiple predictions
all_fv_results = {}
all_cdr_results = {}


# Assigns a local directory for the transformers written results
def assign_local_directory(schedule_identity):
    schedule_directory = root_directory + "\\" + "schedule " + schedule_identity
    p = Path(schedule_directory)
    if p.is_dir():
        pass
    else:
        os.mkdir(schedule_directory)

    directory_files = os.listdir(schedule_directory)
    taken_directories = []
    for file in directory_files:
        if file.find("Result") > -1:
            taken_directories.append(int(file.split()[1]))
    if len(taken_directories) == 0:
        result_number = 1
    else:
        result_number = (max(taken_directories)) + 1
    result_directory = schedule_directory + "\\" + "Result " + str(result_number)
    os.mkdir(result_directory)
    return result_directory


# Writes the transformer configuration and results to a txt file
def write_results(write_directory, written_results_dict):
    text_file = open(write_directory + "\\" + "result.txt", "w")
    text_file.write("TRANSFORMER CONFIGURATION:" + "\n")
    text_file.write("Number of encoder layers: " + written_results_dict["encoder_layers"] + "\n")
    text_file.write("Number of decoder layers: " + written_results_dict["decoder_layers"] + "\n")
    text_file.write("Embedding dimension (d_model): " + written_results_dict["embedding_dim"] + "\n")
    text_file.write("feed forward network width (ffn_width): " + written_results_dict["ffn_width"] + "\n")
    text_file.write("number of attention heads (in each multi-head attention layer): " +
                    written_results_dict["num_heads"] + "\n")
    text_file.write("Input vocab size: " + written_results_dict["input_vocab_size"] + "\n")
    text_file.write("Output vocab size: " + written_results_dict["target_vocab_size"] + "\n")
    text_file.write("Dropout rate: " + written_results_dict["dropout_rate"] + "\n")
    text_file.write("Number of Epochs: " + str(written_results_dict["Epochs"]) + "\n")
    text_file.write("Input sequence length: " + written_results_dict["input_seq_length"] + "\n")
    text_file.write("Output sequence length: " + written_results_dict["output_seq_length"] + "\n")
    text_file.write("\n")

    text_file.write("DATASET: " + "\n")
    text_file.write("Dataset identity: " + written_results_dict["dataset_identity"] + "\n")
    text_file.write("Number of training samples: " + written_results_dict["train_ds_size"] + "\n")
    text_file.write("Number of validation samples: " + written_results_dict["val_ds_size"] + "\n")
    text_file.write("Number of test samples: " + written_results_dict["test_ds_size"] + "\n")
    text_file.write("\n")

    text_file.write("METRICS: " + "\n")
    for i in range(len(written_results_dict["Epochs"])):
        text_file.write("Test accuracy at epoch " + str(written_results_dict["Epochs"][i]) + ": "
                        + written_results_dict["test_accuracy"][i] + "\n")
        text_file.write("Test loss at epoch " + str(written_results_dict["Epochs"][i]) + ": "
                        + written_results_dict["test_loss"][i] + "\n")
    text_file.write("\n")

    text_file.write("PREDICTIONS:" + "\n")
    text_file.write("Ancestor sequence used: " + written_results_dict["ancestor_sequence"] + "\n")
    for i in range(len(written_results_dict["Epochs"])):
        text_file.write("PREDICTION AT EPOCH: " + str(written_results_dict["Epochs"][i]) + ": " + "\n")
        text_file.write("Predicted descendant DNA: " + written_results_dict["predicted_descendant_design"][i] + "\n")
        text_file.write("\n")


# Writes images to the directory
def write_image(image, image_identity, write_directory, epoch="", layer_number=""):
    if image_identity == "attention_head":
        taken_identities = []
        directory_files = os.listdir(write_directory)
        for file in directory_files:
            if file.find("epoch_" + str(epoch) + "_" + "layer" + "_" + layer_number + "_" + "attention_head") > -1:
                taken_identities.append(int(file.split()[1][:-4]))
        if len(taken_identities) == 0:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " 1" + ".png")
        else:
            image.savefig(write_directory + "\\" + "epoch_" + str(
                epoch) + "_" + "layer" + "_" + layer_number + "_" + image_identity + " " +
                          str(max(taken_identities) + 1) + ".png")
    else:
        image.savefig(write_directory + "\\" + "epoch_" + str(epoch) + "_" + image_identity + ".png")
