# Generating a sequence logo to visualise the confidence with which predicted amino acid residues are chosen
import logomaker as lm
import pandas as pd
import matplotlib.pyplot as plt
import Write_Results


def create_probability_logo(prob_dict, vocabulary, write_directory, epoch):
    try:
        vocabulary_upper = []
        for token in vocabulary:
            vocabulary_upper.append(token.upper())
        df_descendant = pd.DataFrame(columns=vocabulary_upper[2:], dtype=int)  # "2:" to exclude UNK and pad tokens
        df_descendant_dict_keys = (list(prob_dict.keys()))

        for key in df_descendant_dict_keys:
            for column in df_descendant.columns:
                if prob_dict[key][1] == column.lower():
                    df_descendant.loc[int(key), column] = prob_dict[key][0]
                else:
                    df_descendant.loc[int(key), column] = 0
        df_descendant.index = range(1, 787)  # Sequence length 786

        logo_descendant = lm.Logo(df_descendant, font_name='Arial Rounded MT Bold', color_scheme="skylign_protein", figsize=(45, 5))
        logo_descendant.ax.set_xlabel('Position', fontsize=14)
        logo_descendant.ax.set_ylabel("Probability", labelpad=1, fontsize=14)
        Write_Results.write_image(logo_descendant.fig, "logo_descendant", write_directory, epoch)

        plt.figure()

    except IndexError:
        print("An error was encountered during creation of the probability logo")
        pass


