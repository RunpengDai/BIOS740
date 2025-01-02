import glob
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import os
import yaml
import cv2
from matplotlib import pyplot as plt
import nibabel as nib
import re
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _get_image_by_id(images_folder, id, file_suffix="png"):
    image_path = os.path.join(images_folder, f"{id}.{file_suffix}")
    image = read_image(image_path, ImageReadMode.RGB)
    return image


def _get_mask_by_id(masks_folder, id, file_suffix="png"):
    mask_path = os.path.join(masks_folder, f"{id}.{file_suffix}")
    mask = read_image(mask_path, ImageReadMode.GRAY)
    return mask


def _process_data(ED_file, ES_file, out_folder, cnt, mask=False, slice_failed=None):
    # Check file name format
    if not mask:
        pattern = r"_frame(\d+)\.nii\.gz"
    else:
        pattern = r"_frame(\d+)_gt\.nii\.gz"
    ED_match = re.search(pattern, ED_file)
    ES_match = re.search(pattern, ES_file)
    if not ED_match or not ES_match:
        raise ValueError("File name incorrect")

    ED_data = nib.load(ED_file).get_fdata()
    ES_data = nib.load(ES_file).get_fdata()
    if ED_data.shape != ES_data.shape:
        raise ValueError("ED frame and ES frame have mismatched shapes")

    patient_name = ED_file.split("/")[-1].split("_")[0]
    if mask:
        slice_failed = []

    for s in range(ED_data.shape[2]):  # iterate through slices
        if not mask and slice_failed is not None and s in slice_failed:
            continue

        # Exclude slices with no annotation, which usually happens on the first or last few slices
        if mask and (
            len(np.unique(ED_data[:, :, s])) == 1
            or len(np.unique(ES_data[:, :, s])) == 1
        ):
            slice_failed.append(s)
            continue

        ED_image = os.path.join(out_folder, f"{cnt}.png")
        cnt += 1
        ES_image = os.path.join(out_folder, f"{cnt}.png")
        cnt += 1

        plt.imsave(ED_image, ED_data[:, :, s], cmap="gray")
        plt.imsave(ES_image, ES_data[:, :, s], cmap="gray")

    if mask and len(slice_failed) > 0:
        print(f"Slices {slice_failed} failed for {patient_name}")

    return cnt, slice_failed


def _process_batch(
    data_ids,
    batch_start,
    batch_end,
    images_folder,
    masks_folder,
    image_transform=None,
    mask_transform=None,
):
    batch_images = []
    batch_masks = []

    for i in range(batch_start, batch_end):
        data_id = data_ids[i]
        image = _get_image_by_id(images_folder, data_id)
        mask = _get_mask_by_id(masks_folder, data_id)

        if image_transform:
            image = image_transform(image)
            image = (image - image.min()) / (image.max() - image.min())
        if mask_transform:
            mask = mask_transform(mask)
            mask = (mask - mask.min()) / (mask.max() - mask.min())

        image = image.numpy()
        mask = mask.numpy()

        batch_images.append(image)
        batch_masks.append(mask)

    return np.array(batch_images), np.array(batch_masks)


def _save_batch(
    data_ids,
    mode,
    images_folder,
    masks_folder,
    np_out_folder,
    training_len,
    val_len,
    testing_len,
    batch_size,
    image_transform,
    mask_transform,
):
    if mode == "train":
        print("Saving training data as batched numpy files")
        data_len = training_len
    elif mode == "val":
        print("Saving validation data as batched numpy files")
        data_len = val_len
    elif mode == "test":
        print("Saving testing data as batched numpy files")
        data_len = testing_len
    else:
        raise ValueError("mode must be train, val or test")
    num_batches = data_len // batch_size
    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        if mode == "val":
            batch_start += training_len
            batch_end += training_len
        elif mode == "test":
            batch_start += training_len + val_len
            batch_end += training_len + val_len

        batch_images, batch_masks = _process_batch(
            data_ids,
            batch_start,
            batch_end,
            images_folder,
            masks_folder,
            image_transform,
            mask_transform,
        )
        np.save(f"{np_out_folder}/X_{mode}_224x224_batch{batch_idx}", batch_images)
        np.save(f"{np_out_folder}/Y_{mode}_224x224_batch{batch_idx}", batch_masks)

    # tail case
    batch_start = num_batches * batch_size
    batch_end = data_len
    if mode == "val":
        batch_start += training_len
        batch_end += training_len
    elif mode == "test":
        batch_start += training_len + val_len
        batch_end += training_len + val_len

    batch_images, batch_masks = _process_batch(
        data_ids,
        batch_start,
        batch_end,
        images_folder,
        masks_folder,
        image_transform,
        mask_transform,
    )
    np.save(f"{np_out_folder}/X_{mode}_224x224_batch{num_batches}", batch_images)
    np.save(f"{np_out_folder}/Y_{mode}_224x224_batch{num_batches}", batch_masks)


def Preprocess_ACDC(
    dataset_folder,
    images_folder="../data/images",
    masks_folder="../data/masks",
    np_out_folder="../data/np_data",
    input_size=224,
    batch_size=20,
    file_suffix="png",
):
    """
    Preprocess the ACDC dataset. We will extract all ED and ES frames from each patient and save them as images.
    The training set will contain 100 patients, the validation set will contain 20 patients, and the test set will contain the remaining 30 patients.
    """
    # remove existing images and masks
    if os.path.exists(images_folder):
        shutil.rmtree(images_folder, ignore_errors=True)
    if os.path.exists(masks_folder):
        shutil.rmtree(masks_folder, ignore_errors=True)
    if os.path.exists(np_out_folder):
        shutil.rmtree(np_out_folder, ignore_errors=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(np_out_folder, exist_ok=True)

    # Records the number of images
    cnt_image = 0
    cnt_mask = 0
    nums = {}
    for patient_folder in tqdm(os.listdir(dataset_folder)):
        patient_name = patient_folder
        patient_folder_path = os.path.join(dataset_folder, patient_folder)
        # skip abnormal data
        if patient_name == "patient090":
            continue

        if os.path.isdir(patient_folder_path):
            # Iterate through files in each patient folder
            info_path = os.path.join(patient_folder_path, "Info.cfg")
            with open(info_path) as info_file:
                info = info_file.readlines()
            # ED is the end-diastolic time, ES is the end-systolic time of the heart
            ED = int(info[0][4:])
            ES = int(info[1][4:])

            # Process nifti files
            ED_mask_file = os.path.join(
                patient_folder_path,
                f"{patient_name}_frame{str(ED).zfill(2)}_gt.nii.gz",
            )
            ES_mask_file = os.path.join(
                patient_folder_path,
                f"{patient_name}_frame{str(ES).zfill(2)}_gt.nii.gz",
            )
            if not os.path.exists(ED_mask_file):
                print(f"Missing ED ground truth {ED_mask_file}")
                continue
            if not os.path.exists(ES_mask_file):
                print(f"Missing ES ground truth {ES_mask_file}")
                continue

            cnt_mask, slice_failed = _process_data(
                ED_mask_file, ES_mask_file, masks_folder, cnt_mask, mask=True
            )

            ED_file = os.path.join(
                patient_folder_path, f"{patient_name}_frame{str(ED).zfill(2)}.nii.gz"
            )
            ES_file = os.path.join(
                patient_folder_path, f"{patient_name}_frame{str(ES).zfill(2)}.nii.gz"
            )
            if not os.path.exists(ED_file):
                print(f"Missing ED {ED_file}")
                continue
            if not os.path.exists(ES_file):
                print(f"Missing ES {ES_file}")
                continue
            cnt_image, _ = _process_data(
                ED_file,
                ES_file,
                images_folder,
                cnt_image,
                mask=False,
                slice_failed=slice_failed,
            )

        assert cnt_image == cnt_mask, "Number of images and labels do not match"
        if patient_name == "patient100":
            nums["train"] = cnt_image
        elif patient_name == "patient120":
            nums["val"] = cnt_image - nums["train"]
        elif patient_name == "patient150":
            nums["test"] = cnt_image - nums["train"] - nums["val"]

    image_transform = transforms.Compose(
        [
            transforms.Resize(
                size=[input_size, input_size],
                interpolation=transforms.functional.InterpolationMode.BILINEAR,
            ),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                size=[input_size, input_size],
                interpolation=transforms.functional.InterpolationMode.NEAREST,
            ),
        ]
    )

    # Save all generated pngs as batched numpy files
    images = glob.glob(f"{images_folder}/*.{file_suffix}")
    data_ids = [d.split(f".{file_suffix}")[0].split("/")[-1] for d in images]

    _save_batch(
        data_ids,
        "train",
        images_folder,
        masks_folder,
        np_out_folder,
        nums["train"],
        nums["val"],
        nums["test"],
        batch_size,
        image_transform,
        mask_transform,
    )
    _save_batch(
        data_ids,
        "val",
        images_folder,
        masks_folder,
        np_out_folder,
        nums["train"],
        nums["val"],
        nums["test"],
        batch_size,
        image_transform,
        mask_transform,
    )
    _save_batch(
        data_ids,
        "test",
        images_folder,
        masks_folder,
        np_out_folder,
        nums["train"],
        nums["val"],
        nums["test"],
        batch_size,
        image_transform,
        mask_transform,
    )

    return nums


class ACDC_Dataset(Dataset):
    def __init__(
        self,
        mode,
        data_dir,
        len_dict,
        batch_size=20,
        one_hot=True,
        num_classes=None,
        silent=True,
    ):
        # pre-set variables
        self.data_dir = data_dir
        self.batch_size = batch_size

        # input parameters
        self.mode = mode
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.images = None
        self.masks = None

        self.training_len = len_dict["train"]
        self.val_len = len_dict["val"]
        self.testing_len = len_dict["test"]

        if not silent:
            if self.mode == "train":
                print("There are {} images in training set".format(self.training_len))
            elif self.mode == "val":
                print(
                    "There are {} images in validation set".format(self.val_len)
                )
            elif self.mode == "test":
                print("There are {} images in testing set".format(self.testing_len))
            else:
                raise ValueError("mode should be one of 'train', 'val', 'test'")

    def __len__(self):
        if self.mode == "train":
            return self.training_len
        elif self.mode == "val":
            return self.val_len
        elif self.mode == "test":
            return self.testing_len
        else:
            raise ValueError("mode should be one of 'train', 'val', 'test'")

    def __getitem__(self, idx):
        batch_id = idx // self.batch_size
        data_in_batch_id = idx % self.batch_size
        data_id = idx
        self.images = torch.tensor(
            np.load(f"{self.data_dir}/X_{self.mode}_224x224_batch{batch_id}.npy")
        )
        self.masks = torch.tensor(
            np.load(f"{self.data_dir}/Y_{self.mode}_224x224_batch{batch_id}.npy")
        )

        image = self.images[data_in_batch_id]
        mask = self.masks[data_in_batch_id]

        if self.one_hot:
            if self.num_classes is None or self.num_classes == 2:
                mask_onehot = F.one_hot(
                    torch.squeeze(mask).to(torch.int64)
                )  # torch.squeeze can remove 1-dim, e.g. A*1*B -> A*B
                mask_onehot = torch.moveaxis(mask_onehot, -1, 0).to(
                    torch.float
                )  # torch.moveaxis with (-1,0) move the last index to first position
            else:
                mask = mask * (self.num_classes - 1)  # make sure one_hot works
                mask = torch.squeeze(mask).to(torch.int64)
                mask_onehot = F.one_hot(mask, num_classes=self.num_classes)
                mask_onehot = torch.moveaxis(mask_onehot, -1, 0).to(torch.float)

        sample = {
            "image": image,
            "mask": mask,
            "mask_onehot": mask_onehot,
            "id": data_id,
        }
        return sample


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config


def make_serializeable_metrics(computed_metrics):
    res = {}
    for k, v in computed_metrics.items():
        res[k] = float(v.cpu().detach().numpy())
    return res


def skin_plot(image, mask, pred):
    image = np.array(image)
    mask = np.array(mask)
    pred = np.array(pred)
    edge_pred = cv2.Canny(pred, 100, 255)
    contours_test, _ = cv2.findContours(
        edge_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    edged_mask = cv2.Canny(mask, 100, 255)
    contours_mask, _ = cv2.findContours(
        edged_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt_pred in contours_test:
        cv2.drawContours(image, [cnt_pred], -1, (255, 0, 0), 1)
    for cnt_mask in contours_mask:
        cv2.drawContours(image, [cnt_mask], -1, (0, 255, 0), 1)
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=400):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
