import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import os

"""
This file is directly copied from the repository
https://github.com/schaugf/HEnorm_python
"""


def normalize_staining(img, img_id=None, saveFile=None, Io=240, alpha=1, beta=0.15, HEstain=False):
    ''' Normalize staining appearence of H&E and H-DAB stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        HEstain: True, normalizes H&E staining
                 False, normalizes H-DAB staining

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    if HEstain:
        HERef = np.array([[0.651, 0.2159],
                        [0.701, 0.8012],
                        [0.29, 0.5581]])


    else:
        HERef = np.array([[0.651, 0.269],
                        [0.701, 0.568],
                        [0.29, 0.778]])


    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float)+1)/Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]
    if ODhat.size == 0:
        return None, None, None

    # compute eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    except:
        return None, None, None

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        # change the ending of the file names to get other formats
        Image.fromarray(Inorm).save(Path(saveFile, img_id + '_normalized.jpg'))
        Image.fromarray(H).save(Path(saveFile, img_id + '_normalized_H.jpg'))
        Image.fromarray(E).save(Path(saveFile, img_id + '_normalized_E.jpg'))

    return Inorm, H, E




