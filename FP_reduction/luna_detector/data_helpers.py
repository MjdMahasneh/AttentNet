import SimpleITK as sitk
import numpy as np
import csv
from datetime import datetime
from scipy import ndimage
import os


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def worldToVoxelCoord(worldCoord, origin, spacing):
    strechedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = strechedVoxelCoord / spacing
    return voxelCoord


def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


def mixArrays(firstArray, secondArray):
    resultList = []
    for i in range(2 * len(firstArray)):
        if i % 2 == 0:
            resultList.append(firstArray[int(i / 2)])
        else:
            resultList.append(secondArray[int(i / 2)])
    return resultList


def mixArraysNumpy2(firstArray, secondArray, zDimension, xyDimension):
    result = np.empty([2 * len(firstArray), zDimension, xyDimension, xyDimension])
    for i in range(2 * len(firstArray)):
        if i % 2 == 0:
            result[i] = firstArray[int(i / 2)]
        else:
            result[i] = secondArray[int(i / 2)]
    return result


def padding(array, zz, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :param yy: desired depth
    :return: padded array
    """

    z = array.shape[0]
    h = array.shape[1]
    w = array.shape[2]

    c = (zz - z) // 2
    cc = zz - c - z

    b = (yy - h) // 2
    bb = yy - b - h

    a = (xx - w) // 2
    aa = xx - a - w

    return np.pad(array, pad_width=((c, cc), (b, bb), (a, aa)), mode='constant', constant_values=0.)


def get_patch(newVolume, cand, numpyOrigin, RESIZE_SPACING, voxelWidthZ, voxelWidthXY):
    voxelWorldCoor = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
    newGeneratedCoor = worldToVoxelCoord(voxelWorldCoor, numpyOrigin, RESIZE_SPACING)

    patch = newVolume[int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
            int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
            int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]

    if np.any(patch.shape != (voxelWidthZ, voxelWidthXY, voxelWidthXY)):
        print('padding...')
        try:
            patch = padding(patch, voxelWidthZ, voxelWidthXY, voxelWidthXY)
        except Exception as e:
            print('ops!! {}'.format(e))

    return patch


def crop_patch(vol, voxelWidthZ, voxelWidthXY):
    z, y, x = np.shape(vol)
    z_prime, y_prime, x_prime = voxelWidthZ, voxelWidthXY, voxelWidthXY
    patch = vol[int(z / 2 - z_prime / 2):int(z / 2 + z_prime / 2),
            int(y / 2 - y_prime / 2):int(y / 2 + y_prime / 2),
            int(x / 2 - x_prime / 2):int(x / 2 + x_prime / 2)]

    return patch


def zoom_patch(patch, resized_voxelWidthZ, resized_voxelWidthXY):
    try:
        if np.shape(patch) != (resized_voxelWidthZ, resized_voxelWidthXY, resized_voxelWidthXY):
            zoomFactor = [resized_voxelWidthZ / float(np.shape(patch)[0]),
                          resized_voxelWidthXY / float(np.shape(patch)[1]),
                          resized_voxelWidthXY / float(np.shape(patch)[2])]
            patch = ndimage.zoom(patch, zoom=zoomFactor)
    except Exception as e:
        print('ops! Error in zoom_patch() {}'.format(str(e)))
        patch = np.zeros((resized_voxelWidthZ, resized_voxelWidthXY, resized_voxelWidthXY))
    return patch


def get_img_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
