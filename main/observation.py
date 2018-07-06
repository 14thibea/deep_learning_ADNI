import torch
import numpy as np
import nibabel as nib
import os
from os import path
from autoencoder_training import AdaptativeAutoEncoder
from data_loader import MriBrainDataset, ToTensor
import pandas as pd


def ranking_list(input_list):
    """
    Returns the rank of each element of the list

    :param input_list: a list of elements
    :return: output_list: the corresponding ranks
    """
    indices = list(range(len(input_list)))
    indices.sort(key=lambda x: input_list[x])
    output_list = [0] * len(indices)
    for i, x in enumerate(indices):
        output_list[x] = i
        
    return output_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help='path to the parameters of the model')
    parser.add_argument("data_path", type=str,
                        help="path data for visualisation")
    parser.add_argument("caps_path", type=str,
                        help='path to the CAPS folder')
    parser.add_argument("--max", type=int, default=10,
                        help="Number of feature maps that are saved")
    parser.add_argument("--filters", type=int, default=150,
                        help="Number of layers in the autoencoder")

    args = parser.parse_args()

    device = torch.device("cuda")
    model = AdaptativeAutoEncoder(args.filters).to(device=device)
    model.load_state_dict(torch.load(args.model_path))
    bias = model.encode.bias.cpu().detach().numpy()
    weights = model.encode.weight.cpu().detach().numpy()
    ranks = ranking_list(bias)

    results_path = '/'.join(args.model_path.split('/')[:-1:]) + '/visualizations'
    if not path.exists(results_path):
        os.makedirs(results_path)

    dataset = MriBrainDataset(args.data_path, args.caps_path, ToTensor())
    columns = ["diagnosis"]
    for i in range(len(ranks)):
        columns.append('hidden-' + str(i))
    hidden_df = pd.DataFrame(index=dataset.subjects_list(), columns=columns)

    for sub in range(len(dataset)):
        sample = dataset[sub]
        img_tensor = sample['image'].view(1, 1, 121, 145, 121).cuda()
        name = sample['name']
        diagnosis = 'diag-' + str(sample['diagnosis'])
        hidden_df.loc[name, 'diagnosis'] = diagnosis
        sub_path = path.join(results_path, diagnosis, name)
        if not path.exists(sub_path) and sub < args.max:
            os.makedirs(sub_path)

        output, hidden, downsampled = model(img_tensor)

        if sub < args.max:
            # Saving downsampled image
            downsampled_np = downsampled.cpu().numpy()
            final_image = nib.Nifti1Image(downsampled_np[0, 0], affine=2 * np.eye(4))
            nib.save(final_image, path.join(sub_path, 'downsampled_brain.nii'))

        # Saving hidden layer
        hidden_np = hidden.cpu().detach().numpy()
        for i, rank in enumerate(ranks):
            final_image = nib.Nifti1Image(hidden_np[0, i], affine=2 * np.eye(4))
            weight = weights[i]
            non_empty = np.where(hidden_np[0, i] != 0)[0]
            if len(non_empty) > 0:
                hidden_df.loc[name, 'hidden-' + str(i)] = 1
                if sub < args.max:
                    nib.save(final_image, path.join(sub_path, 'hidden-' + str(i) + '.nii'))
            else:
                hidden_df.loc[name, 'hidden-' + str(i)] = 0

        if sub < args.max:
            # Saving output (reconstruction of downsampled image)
            output_np = output.cpu().detach().numpy()
            final_image = nib.Nifti1Image(output_np[0, 0], affine=2 * np.eye(4))
            nib.save(final_image, path.join(sub_path, 'output_brain.nii'))

    hidden_df.to_csv(path.join(results_path, 'hidden.tsv'), sep='\t')
