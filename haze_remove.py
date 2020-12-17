import cv2
import numpy as np
from scipy.ndimage.filters import minimum_filter


# step 1: compute the dark channel
def dark_channel(im, patch_size=15):
    dark = minimum_filter(im, patch_size, mode='nearest')
    dark = np.min(dark, axis=2)
    return dark

# step 2: compute airlight
def atmospheric_light(im, dark, mean=True):
    # We first pick the top 0.1% brightest pixels in the dark channel.
    # Among these pixels, the pixels with highest intensity in the input 
    # image I is selected as the atmospheric light
    flat = dark.flatten()
    num = flat.shape[0] >> 10 # same as / 1024
    assert num >= 1
    indice = flat.argsort()[-num:]
    cols = dark.shape[1]
    xys = [(index // cols, index % cols) for index in indice]
    # In paper, author haven't say we should use average 
    # but in practice, average value yield better result
    if mean:
        points = np.array([im[xy] for xy in xys])
        airlight = points.mean(axis=0)
        return airlight
    xys = sorted(xys, key=lambda xy: sum(im[xy]), reverse=True)
    xy = xys[0]
    airlight = im[xy]
    return airlight

# step 3: Estimating the Transmission and refine it
def estimate_transmission(im, airlight, patch_size=15):
    normal = im / airlight
    tx = 1 - 0.95 * dark_channel(normal, patch_size)
    return tx

def refine_transmission():
    # TODO: using Guided filter
    pass

# step 4: Recovering the Scene Radiance
def recover_scene(im, airlight, tx, t0=0.1):
    mtx = np.where(tx > t0, tx, t0)
    res = np.zeros_like(im, dtype=im.dtype)
    for i in range(3):
        c = (im[:, :, i] - airlight[i]) / mtx + airlight[i]
        c = np.where(c < 0, 0, c) 
        c = np.where(c > 255, 255, c)
        res[:, :, i] = c
    return res

def show(*ims):
    for (i, im) in enumerate(ims):
        cv2.imshow(f"im{i}", im) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        img = 'image/haze2.jpg'
    im = cv2.imread(img)
    patch_size = 15
    dark = dark_channel(im, patch_size)
    airlight = atmospheric_light(im, dark, True)
    tx = estimate_transmission(im, airlight, patch_size)
    dehaze = recover_scene(im, airlight, tx, 0.1)
    show(im, dark, tx, dehaze)