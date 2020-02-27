import glob
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

import radial_profile

EPSILON = 1e-8
# interpolation number
N = 300
# number of samples from each type (real and deepfake)
NUM_ITER = 9000


def skip(it, n):
    for i, x in enumerate(it):
        if i < n:
            continue
        yield x


def process_image(filename):
    img = cv2.imread(filename, 0)
    # Crop image to get inner face
    h = int(img.shape[0] / 3)
    w = int(img.shape[1] / 3)
    img = img[h:-h, w:-w]
    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += EPSILON
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radial_profile.azimuthal_average(magnitude_spectrum)
    # Interpolation
    points = np.linspace(0, N, num=psd1D.size)
    xi = np.linspace(0, N, num=N)
    interpolated = griddata(points, psd1D, xi, method="cubic")
    # Normalization
    interpolated /= interpolated[0]
    return interpolated


def process_data(name, label):
    psd1D_total = np.zeros([NUM_ITER, N])
    label_total = np.zeros([NUM_ITER])
    psd1D_org_mean = np.zeros(N)
    psd1D_org_std = np.zeros(N)
    cont = 0

    print("> " + name)
    for filename in skip(glob.glob(name + "*.jpg"), 900):
        interpolated = process_image(filename)

        psd1D_total[cont, :] = interpolated
        label_total[cont] = label
        cont += 1

        if cont == NUM_ITER:
            break

    for x in range(N):
        psd1D_org_mean[x] = np.mean(psd1D_total[:, x])
        psd1D_org_std[x] = np.std(psd1D_total[:, x])

    return psd1D_total, label_total, psd1D_org_mean, psd1D_org_std


def main():
    psd1D_totals, label_totals, y, error = zip(
        *[
            process_data(name, label)
            for name, label in (
                ("../../data_train_wild_400/train/fake/", 0),
                ("../../data_train_wild_400/train/real/", 1),
            )
        ]
    )

    psd1D_total_final = np.concatenate(psd1D_totals, axis=0)
    label_total_final = np.concatenate(label_totals, axis=0)

    data = {}
    data["data"] = psd1D_total_final
    data["label"] = label_total_final

    output = open("features.pkl", "wb")
    pickle.dump(data, output)
    output.close()

    print("DATA Saved")

    x = np.arange(0, N, 1)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(x, y[0], alpha=0.5, color="red", label="fake", linewidth=2.0)
    ax.fill_between(x, y[0] - error[0], y[0] + error[0], color="red", alpha=0.2)

    ax.plot(x, y[1], alpha=0.5, color="blue", label="real", linewidth=2.0)
    ax.fill_between(x, y[1] - error[1], y[1] + error[1], color="blue", alpha=0.2)

    ax.set_title("Satistics 1D Power Spectrum", size=20)
    plt.xlabel("Spatial Frequency", fontsize=20)
    plt.ylabel("Power Spectrum", fontsize=20)
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=20)
    ax.legend(loc="best", prop={"size": 20})
    plt.show()


if __name__ == "__main__":
    main()
