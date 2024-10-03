import numpy as np
import matplotlib.pyplot as plt
import math


def visualize_avg_multi_view(vol):
    '''visualizes the average of the 3 veiws'''

    vol_shape = np.shape(vol)

    avg_vol = (vol[:, :, vol_shape[2] // 2] + vol[:, vol_shape[1] // 2, :] + vol[vol_shape[0] // 2, :, :].T) / 3.0
    plt.imshow(avg_vol)
    plt.title('Averaged vol over Sagittal, Coronal, and Axial')

    plt.show()

    return


def visualize_multi_view(vol):
    vol_shape = np.shape(vol)

    a1 = plt.subplot(2, 2, 1)

    plt.imshow(vol[:, :, vol_shape[2] // 2])

    plt.title('Sagittal view')

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(vol[:, vol_shape[1] // 2, :])

    plt.title('Coronal view')

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(vol[vol_shape[0] // 2, :,
               :])

    plt.title('Axial view')

    a3 = plt.subplot(2, 2, 4)
    plt.imshow((vol[:, :, vol_shape[2] // 2] + vol[:, vol_shape[1] // 2, :] + vol[vol_shape[0] // 2, :,
                                                                              :].T) / 3.0)

    plt.title('Averaged vol over Sagittal, Coronal, and Axial')

    plt.show()

    return


def visualize_multiple_3D_images(vol1, vol2, vol3):
    vol1_shape = np.shape(vol1)
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(vol1[vol1_shape[0] // 2, :, :])

    plt.title('vol-1 view')

    vol2_shape = np.shape(vol2)
    a2 = plt.subplot(2, 2, 2)
    plt.imshow(vol2[vol2_shape[0] // 2, :, :])

    plt.title('vol-2 view')

    vol3_shape = np.shape(vol3)
    a3 = plt.subplot(2, 2, 3)
    plt.imshow(vol3[vol3_shape[0] // 2, :,
               :])

    plt.title('vol-3 view')

    plt.show()

    return


def visualize_vol(vol, title='visualize_vol image'):
    vol_shape = np.shape(vol)
    plt.imshow(vol[vol_shape[0] // 2, :, :])
    plt.title(title)
    plt.show()

    return


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1)
    ax.images[0].set_array(volume[ax.index])


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1)
    ax.images[0].set_array(volume[ax.index])


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def multi_slice_viewer(volume, title_txt='interactive MatPlotLib'):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    print('NOTE for user : Use J and K keys to navigate between the volume slices')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.title(title_txt)
    plt.show()


def mosaic_3D(vol, vol_shape, starting_slc=0, step=20, imgs_per_row=6, res=100):
    '''visualize multiple 2D (mosaic) slices of 3D volumetric image
        parmeters :
        vol : 3D volumatric image file
        starting_slc : slice to start from (first slice to view from vol). Default is 0.
        step = step to taken between images (e.g. second 2D slice to be viewed will be vol[starting_slc + step]). Default is 20,
        vol_shape.
        imgs_per_row = number of images to be viewed per row. Default is 6.
        res = resolution of figure. Default is 100'''

    slice_idxes = range(starting_slc, vol_shape[0] - starting_slc, step)

    rows, cols = math.ceil(len(slice_idxes) / imgs_per_row), imgs_per_row
    fig = plt.figure(figsize=(rows, cols), dpi=res)
    for i in range(1, cols * rows + 1):
        if i > len(slice_idxes):
            break

        slice_2D = vol[slice_idxes[i - 1], :, :].T
        fig.add_subplot(rows, cols, i)
        plt.imshow(slice_2D)
        plt.title('Slice ' + str(slice_idxes[i - 1]))

    plt.show()
    return
