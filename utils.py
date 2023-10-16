import os
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import skimage.color as color
import cv2
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_image_path(data_folder):
    """
    Return the paths of source and target images in the given folder
    """
    source_path, target_path = None, None
    for filename in os.listdir(data_folder):
        if "source" in filename:
            source_path = os.path.join(data_folder, filename)
        elif "target" in filename:
            target_path = os.path.join(data_folder, filename)
        else:
            raise FileExistsError("Please make sure that there is a target image and a source image, with a note in the filename.")
    return source_path, target_path


def generate_displacement_field_path(data_folder):
    """
    Return the paths of the displacement fields in the given folder
    """
    displacement_field_paths = list()
    for filename in os.listdir(data_folder):
        if "displacement_field" in filename:
            displacement_field_paths.append(os.path.join(data_folder,filename))

    if len(displacement_field_paths) > 0:
        return displacement_field_paths
    else:
        raise UserWarning("No displacement field file in the given folder.")



def boundary_detect(image):
    """
    Detect the image boundary used to automatic crop

    :return:
       x, y: the top left pixel coordinate in the level 0 reference frame
       w, h: width and height, describe the specific region size
    """
    # detect boundaries
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    ret, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # choose the boundary with the maximal area
    idx, max_area = 0, 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        rect_area = w * h
        if rect_area > max_area:
            idx = i
            max_area = rect_area
    x, y, w, h = cv2.boundingRect(contours[idx])

    return x, y, w, h


def load_whole_image(filename, level, device=torch.device("cpu")):
    """
    Return the given level image as a tensor
    """
    img = None

    path = os.path.join(filename, f"level_{level}")
    coord_list = [file[:-4].split("_") for file in os.listdir(path)]
    n_row, n_col = 0, 0
    for coord in coord_list:
        row = int(coord[0]) + 1
        col = int(coord[1]) + 1
        if row > n_row:
            n_row = row
        if col > n_col:
            n_col = col

    for row in range(n_row):
        part = None
        for col in range(n_col):
            if col == 0:
                part = load_image_patch(path, (row, col), device=device)
            else:
                part = torch.cat((part, load_image_patch(path, (row, col), device=device)), dim=1)

        img = part if img is None else torch.cat((img, part), dim=0)

    return img


def load_image_patch(img_path, position, device=torch.device("cpu")):
    """
    Return the image patch at the specified location as a tensor
    """
    pos_y = position[0]
    pos_x = position[1]
    img_patch = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_path, f"{pos_y}_{pos_x}.mha")))
    return torch.FloatTensor(img_patch).to(device)


def load_image_patches(img_path, position, k_neighbors, patch_size, device=torch.device("cpu")):
    """
    Return the image patches at the specified location as a tensor

    :param img_path: str, path of the given level image
    :param position: tuple -- (row, col), the position of the middle patches
    :param k_neighbors: int, the range of neighborhood patches
    :param patch_size: int, size of image patch, be useful for generating the neighbors of edge patches
    """
    center_pos_y = position[0]
    center_pos_x = position[1]

    img_patches = None
    for row in range(center_pos_y - k_neighbors, center_pos_y + k_neighbors + 1):
        part = None
        for col in range(center_pos_x - k_neighbors, center_pos_x + k_neighbors + 1):
            img_patch_path = os.path.join(img_path, f"{row}_{col}.mha")
            if os.path.exists(img_patch_path):
                temp_patch = load_image_patch(img_path, (row, col), device=device)
                if temp_patch.size(2) == 4:
                    temp_patch = temp_patch[:, :, :3]
            else:
                temp_patch = torch.zeros((patch_size, patch_size, 3), dtype=torch.float32).to(device)

            if col == center_pos_x - k_neighbors:
                part = temp_patch
            else:
                height = min(part.size(0), temp_patch.size(0))
                part, temp_patch = part[:height, :, :], temp_patch[:height, :, :]
                part = torch.cat((part, temp_patch), dim=1)

        if img_patches is None:
            img_patches = part
        else:
            width = min(img_patches.size(1), part.size(1))
            img_patches, part = img_patches[:, :width, :], part[:, :width, :]
            img_patches = torch.cat((img_patches, part), dim=0)

    return img_patches


def calculate_new_shape(current_shape, size, mode="min"):
    """
    Code from: https://github.com/MWod/DeepHistReg (a bit modified)
    """
    if mode == "min":
        if current_shape[0] > current_shape[1]:
            divider = current_shape[1] / size
        else:
            divider = current_shape[0] / size
        new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))

    elif mode == "max":
        if current_shape[0] < current_shape[1]:
            divider = current_shape[1] / size
        else:
            divider = current_shape[0] / size
        new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))
    else:
        raise ValueError("Please input the correct resampling mode.")

    return new_shape


def resample_tensor(tensor, new_size, padding_mode='zeros', device=torch.device("cpu")):
    """
    Resample the tensor based on the given size
    Code from: https://github.com/MWod/DeepHistReg
    """
    x_size = new_size[1]
    y_size = new_size[0]
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    resampled_tensor = F.grid_sample(tensor.view(1, 1, tensor.size(0), tensor.size(1)), n_grid, mode='bilinear',
                                     padding_mode=padding_mode, align_corners=False)[0, 0, :, :]
    return resampled_tensor


def center_of_mass(tensor, device=torch.device("cpu")):
    """
    Calculate the center of mass of an image tensor
    Code from: https://github.com/MWod/DeepHistReg
    """
    y_size, x_size = tensor.size(0), tensor.size(1)

    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)

    m00 = torch.sum(tensor).item()
    m10 = torch.sum(gx * tensor).item()
    m01 = torch.sum(gy * tensor).item()

    com_x = m10 / m00
    com_y = m01 / m00
    return com_x, com_y


def compose_transforms(t1, t2, device=torch.device("cpu")):
    """
    Compose two transform matrices
    Code from: https://github.com/MWod/DeepHistReg
    """
    tr1 = torch.zeros((3, 3)).to(device)
    tr2 = torch.zeros((3, 3)).to(device)
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    tr1[2, 2] = 1
    tr2[2, 2] = 1
    result = torch.mm(tr1, tr2)
    return result[0:2, :]


def generate_rotation_matrix(angle, x0, y0):
    """
    Generate a rotation transform matrix
    Code from: https://github.com/MWod/DeepHistReg
    """
    angle = angle * np.pi/180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])
    transform = cm1 @ rotation_matrix @ cm2
    return transform[0:2, :]


def tensor_affine2theta(affine, shape):
    """
    Convert the affine transform matrix (numpy) to the theta transform matrix (pytorch)
    Code from: https://github.com/MWod/DeepHistReg

    :param affine: tensor, affine transform matrix
    :param shape: tuple, image shape
    :return: theta: tensor, theta transform matrix
    """
    h, w = shape[0], shape[1]
    temp = affine
    theta = torch.zeros([2, 3])
    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1]*h/w
    theta[0, 2] = temp[0, 2]*2/w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0]*w/h
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2]*2/h + theta[1, 0] + theta[1, 1] - 1
    return theta


def numpy_affine2theta(affine, shape):
    """
    Convert the affine transform matrix (numpy) to the theta transform matrix (pytorch)

    :param affine: numpy array, affine transform matrix
    :param shape: tuple, image shape
    :return: theta: numpy array, theta transform matrix
    """
    h, w = shape[0], shape[1]
    theta = np.zeros([2, 3])
    theta[0, 0] = affine[0, 0]
    theta[0, 1] = affine[0, 1]*h/w
    theta[0, 2] = affine[0, 2]*2/w + affine[0, 0] + affine[0, 1] - 1
    theta[1, 0] = affine[1, 0]*w/h
    theta[1, 1] = affine[1, 1]
    theta[1, 2] = affine[1, 2]*2/h + affine[1, 0] + affine[1, 1] - 1
    return theta


def tensor_affine_transform(tensor, tensor_transform, device=torch.device("cpu")):
    """
    Affine transformation of the image tensor
    Code from: https://github.com/MWod/DeepHistReg
    """
    if len(tensor.size()) == 2:
        tensor = tensor.view(1, 1, tensor.size(0), tensor.size(1))
    if len(tensor_transform.size()) == 2:
        tensor_transform = tensor_transform.view(-1, tensor_transform.size(0), tensor_transform.size(1))

    assert len(tensor.size()) == 4  # (n_samples, 1, height, width)
    assert len(tensor_transform.size()) == 3  # (n_samples, 2, 3)

    affine_grid = F.affine_grid(tensor_transform, tensor.size(), align_corners=False)
    transformed_tensor = F.grid_sample(tensor, affine_grid, align_corners=False)

    return transformed_tensor


def transform_to_displacement_fields(tensors, tensor_transforms, device=torch.device("cpu")):
    """
    Convert the transform matrix to the displacement field
    Code from: https://github.com/MWod/DeepHistReg

    :param tensors: of size (n_samples, channel, height, width) or (height, width)
    :param tensor_transforms: theta transform matrix, of size (n_samples, 2, 3)
    :return: displacement_fields: tensor, of size (n_samples, 2, height, width)
    """
    if len(tensors.size()) == 2:
        tensors = tensors.view(1, 1, tensors.size(0), tensors.size(1))
    if len(tensor_transforms.size()) == 2:
        tensor_transforms = tensor_transforms.view(-1, tensor_transforms.size(0), tensor_transforms.size(1))

    assert len(tensors.size()) == 4  # (n_samples, 1, height, width)
    assert len(tensor_transforms.size()) == 3  # (n_samples, 2, 3)

    n_samples = tensors.size(0)
    y_size = tensors.size(2)
    x_size = tensors.size(3)

    displacement_fields = torch.Tensor([]).to(device)
    for i in range(n_samples):
        tensor = tensors[i, :, :, :].view(-1, tensors.size(1), y_size, x_size)
        tensor_transform = tensor_transforms[i, :, :].view(-1, 2, 3)

        deformation_field = F.affine_grid(tensor_transform, tensor.size(), align_corners=False)

        gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
        gy = gy.type(torch.FloatTensor).to(device)
        gx = gx.type(torch.FloatTensor).to(device)
        grid_x = (gx / (x_size - 1) - 0.5) * 2
        grid_y = (gy / (y_size - 1) - 0.5) * 2

        u_x = deformation_field[0, :, :, 0] - grid_x
        u_y = deformation_field[0, :, :, 1] - grid_y
        u_x = u_x / 2 * (x_size - 1)
        u_y = u_y / 2 * (y_size - 1)

        displacement_field = torch.cat((u_x.view(1, y_size, x_size), u_y.view(1, y_size, x_size)), dim=0)
        displacement_fields = torch.cat((displacement_fields, torch.unsqueeze(displacement_field, dim=0)), dim=0)

    return displacement_fields


def resample_displacement_fields(displacement_fields, new_size, padding_mode='zeros', device=torch.device("cpu")):
    """
    Resample the displacement fields to the given new size
    """
    if len(displacement_fields.size()) == 3:  # (2, height, width)
        displacement_fields = displacement_fields.view(1, displacement_fields.size(0), displacement_fields.size(1), displacement_fields.size(2))
    assert len(displacement_fields.size()) == 4  # (n_samples, 2, height, width)

    n_samples = new_size[0]

    old_x_size = displacement_fields.size(3)
    old_y_size = displacement_fields.size(2)
    x_size = new_size[3]
    y_size = new_size[2]

    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(n_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(n_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)

    resampled_displacement_fields = F.grid_sample(displacement_fields, n_grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
    resampled_displacement_fields[:, 0, :, :] *= x_size / old_x_size
    resampled_displacement_fields[:, 1, :, :] *= y_size / old_y_size

    return resampled_displacement_fields


def warp_tensors(tensors, displacement_fields, device=torch.device("cpu")):
    """
    Warp the image tensors based on the given displacement fields

    :param tensors: image tensor, of size (n_samples, channel, height, width) or (height, width)
    :param displacement_fields: tensor, of size (_, 2, height, width)
    :return: transformed_tensors: image tensor, of size (n_samples, 1, height, width)
    """
    if len(tensors.size()) == 2:
        tensors = tensors.view(1, 1, tensors.size(0), tensors.size(1))
    if len(displacement_fields.size()) == 3:
        displacement_fields = displacement_fields.view(
            -1, displacement_fields.size(0), displacement_fields.size(1), displacement_fields.size(2)
        )
    assert len(tensors.size()) == 4  # (n_samples, 1, height, width)
    assert len(displacement_fields.size()) == 4  # (n_samples, 2, height, width)

    n_samples = tensors.size(0)
    y_size = tensors.size(2)
    x_size = tensors.size(3)
    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5) * 2
    grid_y = (gy / (y_size - 1) - 0.5) * 2
    n_grid_x = grid_x.view(1, -1).repeat(n_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(n_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)
    displacement_fields = displacement_fields.permute(0, 2, 3, 1)
    u_x = displacement_fields[:, :, :, 0]
    u_y = displacement_fields[:, :, :, 1]
    u_x = u_x / (x_size - 1) * 2
    u_y = u_y / (y_size - 1) * 2
    n_grid[:, :, :, 0] = n_grid[:, :, :, 0] + u_x
    n_grid[:, :, :, 1] = n_grid[:, :, :, 1] + u_y
    transformed_tensors = F.grid_sample(tensors, n_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return transformed_tensors


def tensor_laplacian(tensor, device=torch.device("cpu")):
    """
    Construct Laplacian filter to sharpen the input tensor
    Code from: https://github.com/MWod/DeepHistReg
    """
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).to(device)
    laplacian = F.conv2d(tensor, laplacian_filter.view(1, 1, 3, 3), padding=1) / 9
    return laplacian


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3, device=torch.device("cpu")):
    """
    Code from: https://github.com/MWod/DeepHistReg
    """
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=int((kernel_size / 2)))
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter.to(device)


def get_blur_kernel(kernel_size=3, channels=3, device=torch.device("cpu")):
    """
    Code from: https://github.com/MWod/DeepHistReg
    """
    gaussian_kernel = torch.ones((kernel_size, kernel_size))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=int((kernel_size / 2)))
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter.to(device)


def tensor_edge_extraction(tensor, device=torch.device("cpu")):
    image = np.uint8(tensor.detach().cpu().numpy() * 255)
    tensor_edge = torch.tensor(cv2.Canny(image, 50, 150) / 255, dtype=torch.float32).to(device)
    return tensor_edge


def tensor_shape_detection(tensor, device=torch.device("cpu")):
    image = np.uint8(tensor.detach().cpu().numpy() * 255)
    ret, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    tensor_binary_image = torch.tensor(binary_image / 255, dtype=torch.float32).to(device)
    return tensor_binary_image


def restore_displacement_field(displacement_fields, target_size, padding_tuple, patch_size, device=torch.device("cpu")):
    """
    restore an unfolded displacement field

    :param displacement_fields: unfolded displacement field, tensor, of size (num_patch_sets, 2, patch_size, patch_size)
    :param target_size: the target size of the displacement field, tuple -- (height, width)
    :param padding_tuple: the padding size, tuple -- (padding on height, padding on width)
    :param patch_size: size of the target patch, int
    :return: new_displacement_field: restored displacement field, tensor, of size (1, 2, height, width)
    """
    output_size = (target_size[0] + padding_tuple[0], target_size[1] + padding_tuple[1])
    new_displacement_field = torch.zeros((1, displacement_fields.size(1),) + output_size).to(device)

    col_y, col_x = int(output_size[0] / patch_size), int(output_size[1] / patch_size)
    for j in range(col_y):
        for i in range(col_x):
            current_patch = displacement_fields[j * col_x + i, :, :, :]  # (2, patch_size, patch_size)
            b_y = j * patch_size
            e_y = (j + 1) * patch_size
            b_x = i * patch_size
            e_x = (i + 1) * patch_size
            new_displacement_field[0, :, b_y:e_y, b_x:e_x] = current_patch

    if padding_tuple[0] == 0 and padding_tuple[1] == 0:
        pass
    elif padding_tuple[0] == 0:
        new_displacement_field = new_displacement_field[:, :, :, :-padding_tuple[1]]
    elif padding_tuple[1] == 0:
        new_displacement_field = new_displacement_field[:, :, :-padding_tuple[0], :]
    else:
        new_displacement_field = new_displacement_field[:, :, :-padding_tuple[0], :-padding_tuple[1]]

    return new_displacement_field


def smooth_image_tensor(tensor, kernel_size=5, sigma=1, device=torch.device("cpu")):
    """
    Smooth the input image tensor using the gaussian blur

    :param tensor: image tensor, of size (height, width)
    :param kernel_size: int, the kernel size of the gaussian kernel
    :param sigma: float, sigma of the gaussian kernel
    :return:
    """
    tensor = tensor.view(1, 1, tensor.size(0), tensor.size(1))
    assert len(tensor.size()) == 4  # (1, 1, height, width)

    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, tensor.size(1), device=device)
    smoothed_tensor = gaussian_kernel(tensor)[0, 0, :, :].to(device)
    return smoothed_tensor


def smooth_displacement_fields(displacement_fields, kernel_size=7, gaussian=True, device=torch.device("cpu")):
    """
    Smooth the input displacement field using the gaussian blur

    :param displacement_fields: tensor, of size (n_samples, 2, height, width)
    :param kernel_size: int, the kernel size of the gaussian kernel
    :return:
    """
    if len(displacement_fields.size()) == 3:
        displacement_fields = displacement_fields.view(
            -1, displacement_fields.size(0), displacement_fields.size(1), displacement_fields.size(2)
        )
    assert len(displacement_fields.size()) == 4  # (n_samples, 2, height, width)

    if gaussian:
        kernel = get_gaussian_kernel(kernel_size, 1, displacement_fields.size(1), device=device)
    else:
        kernel = get_blur_kernel(kernel_size, displacement_fields.size(1), device=device)
    return kernel(displacement_fields)


def compose_displacement_fields(u, v, device=torch.device("cpu")):
    """
    Compose the displacement fields
    Code from: https://github.com/MWod/DeepHistReg

    :param u: the first displacement fields, tensor, of size (n_samples, 2, height, width)
    :param v: the second displacement fields, tensor, of size (n_samples, 2, height, width)
    """
    if len(u.size()) == 3:
        u = u.view(-1, u.size(0), u.size(1), u.size(2))
    if len(v.size()) == 3:
        v = v.view(-1, v.size(0), v.size(1), v.size(2))

    size = u.size()
    no_samples = size[0]
    x_size = size[3]
    y_size = size[2]

    gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size))
    gy = gy.type(torch.FloatTensor).to(device)
    gx = gx.type(torch.FloatTensor).to(device)
    grid_x = (gx / (x_size - 1) - 0.5)*2
    grid_y = (gy / (y_size - 1) - 0.5)*2

    u_x_1 = u[:, 0, :, :].view(u.size(0), 1, u.size(2), u.size(3))
    u_y_1 = u[:, 1, :, :].view(u.size(0), 1, u.size(2), u.size(3))
    u_x_2 = v[:, 0, :, :].view(v.size(0), 1, v.size(2), v.size(3))
    u_y_2 = v[:, 1, :, :].view(v.size(0), 1, v.size(2), v.size(3))
    u_x_1 = u_x_1 / (x_size - 1) * 2
    u_y_1 = u_y_1 / (y_size - 1) * 2
    u_x_2 = u_x_2 / (x_size - 1) * 2
    u_y_2 = u_y_2 / (y_size - 1) * 2

    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1))
    n_grid = torch.stack((n_grid_x, n_grid_y), dim=3)

    nv = torch.stack((u_x_2.view(u_x_2.size(0), u_x_2.size(2), u_x_2.size(3)), u_y_2.view(u_y_2.size(0), u_y_2.size(2), u_y_2.size(3))), dim=3)
    t_x = n_grid_x.view(n_grid_x.size(0), 1, n_grid_x.size(1), n_grid_x.size(2))
    t_y = n_grid_y.view(n_grid_y.size(0), 1, n_grid_y.size(1), n_grid_y.size(2))

    added_x = u_x_1 + t_x
    added_y = u_y_1 + t_y
    added_grid = n_grid + nv

    i_u_x = F.grid_sample(added_x, added_grid, padding_mode='border', align_corners=False)
    i_u_y = F.grid_sample(added_y, added_grid, padding_mode='border', align_corners=False)

    indexes = (added_grid[:, :, :, 0] >= 1.0) | (added_grid[:, :, :, 0] <= -1.0) | (added_grid[:, :, :, 1] >= 1.0) | (added_grid[:, :, :, 1] <= -1.0)
    indexes = indexes.view(indexes.size(0), 1, indexes.size(1), indexes.size(2))

    n_x = i_u_x - grid_x
    n_y = i_u_y - grid_y
    # n_x[indexes] = 0.0
    # n_y[indexes] = 0.0
    n_x = n_x / 2 * (x_size - 1)
    n_y = n_y / 2 * (y_size - 1)

    return torch.cat((n_x, n_y), dim=1)


def center_crop_tensor(tensor, cropping_rate, device=torch.device("cpu")):
    """
    Center crop the input tensor (feature map) with a given cropping rate

    :param tensor: feature map, tensor, of size (n_samples, channels, height, width) or (height, width)
    :param cropping_rate: the rate of center-crop operations, int, if cropping_rate=2, the width and height of the
        cropped feature maps become a half of the width and height of the input feature maps
    """
    if len(tensor.size()) == 2:
        tensor = tensor.view(1, 1, tensor.size(0), tensor.size(1))  # (n_samples, 1, height, width)

    if cropping_rate == 1:
        new_tensor = tensor
    else:
        orig_height = tensor.size(2)
        orig_width = tensor.size(3)
        new_height = orig_height // cropping_rate
        new_width = orig_width // cropping_rate

        h_b = (orig_height - new_height) // 2
        h_e = h_b + new_height
        w_b = (orig_width - new_width) // 2
        w_e = w_b + new_width
        new_tensor = tensor[:, :, h_b:h_e, w_b:w_e]

    return new_tensor.to(device)


def create_grid(size, density=100, device=torch.device("cpu")):
    """
    Create a regular grid with a given size
    :param size: tuple of int -- (height, width)
    :param density: density of the generated grid, int
    """
    num_y, num_x = (size[0]) // density, (size[1]) // density
    y, x = np.meshgrid(np.linspace(-2, 2, num_y), np.linspace(-2, 2, num_x))

    plt.figure()
    plt.plot(x, y, linewidth=0.5, color="black")
    plt.plot(x.transpose(), y.transpose(), linewidth=0.5, color="black")
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    plt.close()
    grid_img = color.rgb2gray(np.array(canvas.renderer.buffer_rgba())[:, :, :-1])
    grid_img = torch.from_numpy(grid_img).type(torch.FloatTensor).to(device)
    grid_img = resample_tensor(grid_img, size, device=device)
    return grid_img


def brightness_correction(source, target, correction_coeff=1.0, device=torch.device("cpu")):
    """
    Correct the brightness of the source so that it approximates the target
    :param source: the source images, tensor, of size (n_samples, num_pyramid, height, width)
    :param target: the target images, tensor, of size (n_samples, num_pyramid, height, width)
    :param correction_coeff: the coefficient of brightness correction, float
    :return: corrected_source: the corrected source images, tensor, of size (n_samples, num_pyramid, height, width)
    """
    source_brightness = torch.sum(source) / torch.count_nonzero(source)
    target_brightness = torch.sum(target) / torch.count_nonzero(target)

    brightness_factor = correction_coeff * (target_brightness / source_brightness)
    corrected_source = source ** (1 / brightness_factor)
    return corrected_source


def unfold(tensor, stride=128, patch_size=256, num_pyramid=3, device=torch.device("cpu")):
    """
    Extract sets of multi-magnification (pyramid) patches from the given image tensor

    :param tensor: image tensor, of size (n_samples, channel, height, width) or (height, width)
    :param patch_size: target size of the training patches, int
    :param num_pyramid: number of patches in each set, int
    :return: pyramid_sets: extracted patch sets from an image tensor, tensor, of size (_, num_pyramid, patch_size, patch_size)
    """
    if len(tensor.size()) == 2:
        tensor = tensor.view(1, 1, tensor.size(0), tensor.size(1))  # (n_samples, 1, height, width)

    x_size = tensor.size(3)
    y_size = tensor.size(2)
    input_patch_size = int(patch_size * np.power(2, num_pyramid - 1))

    unfolder = nn.Unfold(input_patch_size, stride=stride)
    b_x = int((input_patch_size - stride) / 2)
    b_y = int((input_patch_size - stride) / 2)
    pad_x = int(stride - x_size % stride) if x_size % stride != 0 else 0
    pad_y = int(stride - y_size % stride) if x_size % stride != 0 else 0
    e_x = pad_x + b_x
    e_y = pad_y + b_y

    new_tensor = F.pad(tensor, (b_x, e_x, b_y, e_y))
    padding_tuple = (pad_y, pad_x)

    new_tensor = unfolder(new_tensor)
    new_tensor = new_tensor.view(new_tensor.size(0), tensor.size(1), input_patch_size, input_patch_size, new_tensor.size(2))
    new_tensor = new_tensor[0].permute(3, 0, 1, 2)  # (num_sets, 1, input_patch_size, input_patch_size)

    return new_tensor, padding_tuple


def create_multi_magnification_patches(tensor, patch_size, magnifications_num, device=torch.device("cpu")):
    """

    :param tensor:
    :param patch_size:
    :param magnifications_num:
    :param device:
    :return:
    """
    num_sets = tensor.size(0)
    mm_patches_sets = None
    for i in range(num_sets):
        current_patch = tensor[i, 0, :, :]  # (height, width)
        mm_patches_set = None
        for j in range(magnifications_num):
            cropped_patch = center_crop_tensor(current_patch, cropping_rate=np.power(2, magnifications_num - 1 - j), device=device)[0, 0, :, :]
            if j == 0:
                mm_patches_set = torch.unsqueeze(cropped_patch, dim=0)
            else:
                resampled_patch = resample_tensor(cropped_patch, (patch_size, patch_size), padding_mode='border', device=device)
                mm_patches_set = torch.cat((mm_patches_set, torch.unsqueeze(resampled_patch, 0)), dim=0)

        if i == 0:
            mm_patches_sets = torch.unsqueeze(mm_patches_set, 0)
        else:
            mm_patches_sets = torch.cat((mm_patches_sets, torch.unsqueeze(mm_patches_set, 0)), dim=0)

    return mm_patches_sets