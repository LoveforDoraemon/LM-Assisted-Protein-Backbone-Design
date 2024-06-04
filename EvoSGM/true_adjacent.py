import pickle
from dataset import ProteinDataset, PaddingCollate
from easydict import EasyDict
import yaml
import torch


def main():
    with open("./config/evosgm_inpainting.yml", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    ss_constraints = True if config.data.num_channels == 9 else False
    dataset = ProteinDataset(
        config.data.dataset_path,
        config.data.attention_path,
        config.data.min_res_num,
        config.data.max_res_num,
        ss_constraints,
    )
    print(f"Length of dataset: {len(dataset)}!")
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    print(f"Length of test_dataset: {len(test_ds)}!")

    test_sampler = torch.utils.data.SequentialSampler(test_ds)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=len(test_ds),
        collate_fn=PaddingCollate(config.data.max_res_num),
    )
    test_iter = iter(test_dl)

    samples_true = next(test_iter)["coords_6d"]

    with open("true_evosgm_adjacent.pkl", "wb") as file:
        pickle.dump(samples_true, file)


if __name__ == "__main__":
    main()
