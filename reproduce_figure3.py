"""Render Figure 3 from the original saved M51a image and reconstructions.

Postprocessing only: nothing is retrained or recomputed from sensor data.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.special import j1

METHODS = ('none', 'wiener', 'rl', 'vc_tv', 'cnn0', 'cnn')
LABELS = ('Conv. + noisy', 'Wiener', 'RL', 'Prox. TV', '1-CNN', '3-CNN')

def recover_inputs(res_dir, data_dir):
    from astropy.io.fits import getdata
    original = np.asarray(getdata(Path(data_dir) / 'M51a.fits'), dtype=float)
    original = original[1200:2224, 1200:2224]
    if original.shape != (1024, 1024):
        raise ValueError(f'Unexpected original crop shape: {original.shape}')
    results = {}
    for method in METHODS:
        path = Path(res_dir) / f'{method}_1024x1024_PSF64_sigma0.010_M51a.mat'
        values = np.asarray(loadmat(path)['Irec'], dtype=float)
        if values.shape != (1024, 1024):
            raise ValueError(f'{path}: unexpected shape {values.shape}')
        results[method] = values
    difference = np.abs(results['cnn'] - results['cnn0'])
    mean_diff, max_diff = float(difference.mean()), float(difference.max())
    if max_diff < 1e-12:
        raise ValueError('3-CNN and 1-CNN archived image arrays are identical')
    return original, results, mean_diff, max_diff

def draw(original, results, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    coords = np.linspace(-1, 1, 64)
    xx, yy = np.meshgrid(coords, coords)
    rr = np.hypot(xx, yy) * np.pi * 5
    psf = np.square(np.divide(j1(rr), rr,
                     out=np.full_like(rr, 0.5), where=rr != 0))
    psf /= psf.sum()

    def crop(image):
        return image[300:-132, 300:-132]

    cropped = crop(original)
    panels = [('Original', cropped / cropped.max()), ('PSF', np.sqrt(psf))]
    panels += [(name, crop(results[method])) for method, name in zip(METHODS, LABELS)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (label, values) in zip(axes.flat, panels):
        if label == 'PSF':
            ax.imshow(values, cmap='gray', interpolation='nearest')
        else:
            ax.imshow(values, cmap='gray', vmin=0.15, vmax=0.8)
        ax.set_title(label)
        ax.axis('off')
    fig.tight_layout()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--res-dir', type=Path, default=Path('res'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('reproduced/figure3.png'))
    args = parser.parse_args()
    original, results, mean_diff, max_diff = recover_inputs(args.res_dir, args.data_dir)
    print(f'Original 3-CNN vs 1-CNN archived image pixel differences: mean={mean_diff:.8g}, max={max_diff:.8g}')
    draw(original, results, args.output)
    print(f'Wrote {args.output}')

if __name__ == '__main__':
    main()
