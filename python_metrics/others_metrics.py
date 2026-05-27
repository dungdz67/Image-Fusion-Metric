import numpy as np
from scipy.signal import fftconvolve
from metrics_dev import print_matrix_stats


# ──────────────────────────────────────────────
# CC  –  Correlation Coefficient
# ──────────────────────────────────────────────
def cc(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Mean Pearson correlation between fused and each source image."""
    def _corr(X, Y):
        Xm, Ym = X - X.mean(), Y - Y.mean()
        return (Xm * Ym).sum() / (np.sqrt((Xm ** 2).sum() * (Ym ** 2).sum()) + 1e-10)
    rAF = _corr(A.astype(np.float64), F.astype(np.float64))
    rBF = _corr(B.astype(np.float64), F.astype(np.float64))
    return float(np.mean([rAF, rBF]))
# endregion



# ---------------------------------------------------------------------------
# QP  –  Quality of Phase Congruency
# ---------------------------------------------------------------------------

def qp(im1: np.ndarray, im2: np.ndarray, fused: np.ndarray) -> float:
    """
    Compute the QP fusion quality metric.

    Parameters
    ----------
    im1 : np.ndarray
        Source image 1 (grayscale, any numeric dtype).
    im2 : np.ndarray
        Source image 2 (grayscale, any numeric dtype).
    fused : np.ndarray
        Fused image (grayscale, any numeric dtype).

    Returns
    -------
    float
        Scalar QP score (product of three correlation coefficients).
    """
    fea_threshold = 0.1

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    fused = fused.astype(np.float64)

    pc1, _, M1, m1 = phase_congruency(im1)
    pc2, _, M2, m2 = phase_congruency(im2)
    pcf, _, Mf, mf = phase_congruency(fused)
    mask = pc1 > pc2
    pc_max = np.where(mask, pc1, pc2)
    M_max  = np.where(mask, M1,  M2)
    m_max  = np.where(mask, m1,  m2)

    mask1 = pc1    > fea_threshold
    mask2 = pc2    > fea_threshold
    mask3 = pc_max > fea_threshold

    result_PC = _correlation_coefficient(pc1, pc2, pc_max, pcf, mask1, mask2, mask3)
    result_M  = _correlation_coefficient(M1,  M2,  M_max,  Mf,  mask1, mask2, mask3)
    result_m  = _correlation_coefficient(m1,  m2,  m_max,  mf,  mask1, mask2, mask3)

    return float(result_PC * result_M * result_m)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Create a normalised 2-D Gaussian kernel (matches MATLAB fspecial)."""
    ax = np.arange(size) - size // 2
    gauss = np.exp(-ax ** 2 / (2 * sigma ** 2))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def _filter2(kernel: np.ndarray, img: np.ndarray) -> np.ndarray:
    """2-D convolution with 'same' boundary (matches MATLAB filter2)."""
    return fftconvolve(img, kernel, mode='same')


def _correlation_coefficient(
    im1: np.ndarray,
    im2: np.ndarray,
    im_max: np.ndarray,
    imf: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    mask3: np.ndarray,
) -> float:
    """
    Compute the local correlation-coefficient quality score for one feature map.
    Corresponds to MATLAB's nested ``correlation_coeffcient`` sub-function.
    """
    window = _gaussian_kernel(11, 1.5)

    C1 = C2 = C3 = 1e-4

    im1   = mask1.astype(np.float64) * im1
    im2   = mask2.astype(np.float64) * im2
    im_max = mask3.astype(np.float64) * im_max

    mu1    = _filter2(window, im1)
    mu2    = _filter2(window, im2)
    muf    = _filter2(window, imf)
    mu_max = _filter2(window, im_max)
    mu1_sq    = mu1    ** 2
    mu2_sq    = mu2    ** 2
    muf_sq    = muf    ** 2
    mu_max_sq = mu_max ** 2

    sigma1_sq   = _filter2(window, im1    * im1)    - mu1_sq
    sigma2_sq   = _filter2(window, im2    * im2)    - mu2_sq
    sigmaMax_sq = _filter2(window, im_max * im_max) - mu_max_sq
    sigmaf_sq   = _filter2(window, imf    * imf)    - muf_sq

    sigma1f   = _filter2(window, im1    * imf) - mu1    * muf
    sigma2f   = _filter2(window, im2    * imf) - mu2    * muf
    sigmaMaxf = _filter2(window, im_max * imf) - mu_max * muf

    res1 = np.zeros_like(im1)
    res2 = np.zeros_like(im1)
    res3 = np.zeros_like(im1)

    idx1 = mask1
    idx2 = mask2
    idx3 = mask3

    res1[idx1] = (sigma1f[idx1] + C1) / (
        np.sqrt(np.abs(sigma1_sq[idx1] * sigmaf_sq[idx1])) + C1
    )
    res2[idx2] = (sigma2f[idx2] + C2) / (
        np.sqrt(np.abs(sigma2_sq[idx2] * sigmaf_sq[idx2])) + C2
    )
    res3[idx3] = (sigmaMaxf[idx3] + C3) / (
        np.sqrt(np.abs(sigmaMax_sq[idx3] * sigmaf_sq[idx3])) + C3
    )

    result = np.maximum(np.maximum(res1, res2), res3)

    A3 = mask3.sum()
    if A3 == 0:
        return 0.0

    return float(result.sum() / A3)


# ---------------------------------------------------------------------------
# Phase congruency
# ---------------------------------------------------------------------------

def _lowpass_filter(shape: tuple, cutoff: float, n: int) -> np.ndarray:
    """
    Butterworth low-pass filter in the frequency domain.
    Matches MATLAB's ``lowpassfilter`` sub-function.
    """
    if not (0 < cutoff <= 0.5):
        raise ValueError("cutoff must be in (0, 0.5]")
    rows, cols = shape
    x = (np.arange(1, cols + 1) - (cols // 2 + 1)) / cols
    y = (np.arange(1, rows + 1) - (rows // 2 + 1)) / rows
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X ** 2 + Y ** 2)
    f = np.fft.fftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))
    return f


def phase_congruency(
    im: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wave_length: int = 3,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.2,
    k: float = 2.0,
    cut_off: float = 0.5,
    g: float = 10.0,
) -> tuple:
    """
    Compute phase congruency of a grayscale image.

    Python port of Kovesi's phasecong3 / Liu's myphasecong3.

    Parameters
    ----------
    im : np.ndarray
        Input grayscale image (float64 preferred).
    nscale : int
        Number of wavelet scales.
    norient : int
        Number of filter orientations.
    min_wave_length : int
        Wavelength of the smallest scale filter.
    mult : float
        Scaling factor between successive filters.
    sigma_on_f : float
        Bandwidth of log-Gabor filter.
    d_theta_on_sigma : float
        Angular bandwidth ratio.
    k : float
        Noise threshold factor.
    cut_off : float
        Frequency spread penalty threshold.
    g : float
        Sharpness of the sigmoid weighting.

    Returns
    -------
    phase_cong : np.ndarray  – overall phase congruency map
    orientation : np.ndarray – dominant orientation map (degrees 0–180)
    M : np.ndarray           – maximum moment (edge strength)
    m : np.ndarray           – minimum moment (corner strength)
    """
    im = im.astype(np.float64)
    rows, cols = im.shape
    epsilon = 1e-4

    theta_sigma = np.pi / norient / d_theta_on_sigma

    image_fft = np.fft.fft2(im)

    zero         = np.zeros((rows, cols))
    total_energy = zero.copy()
    total_sum_an = zero.copy()
    orientation  = zero.copy()
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()

    # Frequency-domain coordinate grids
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    X, Y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(X ** 2 + Y ** 2)
    radius[rows // 2, cols // 2] = 1.0   # avoid log(0)

    theta = np.arctan2(-Y, X)

    radius = np.fft.ifftshift(radius)
    theta  = np.fft.ifftshift(theta)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Low-pass filter and log-Gabor radial components
    lp = _lowpass_filter((rows, cols), 0.45, 15)

    log_gabor = []
    for s in range(nscale):
        wavelength = min_wave_length * mult ** s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_on_f) ** 2))
        lg *= lp
        lg[0, 0] = 0.0
        log_gabor.append(lg)

    # Angular spread components
    spread = []
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sin_theta * np.cos(angl) - cos_theta * np.sin(angl)
        dc = cos_theta * np.cos(angl) + sin_theta * np.sin(angl)
        d_theta = np.abs(np.arctan2(ds, dc))
        spread.append(np.exp(-(d_theta ** 2) / (2 * theta_sigma ** 2)))

    ifft_filter_array = []
    for s in range(nscale):
        filt = log_gabor[s] * spread[0]           # use orientation 0 for noise est.
        ifft_filt = np.real(np.fft.ifft2(filt)) * np.sqrt(rows * cols)
        ifft_filter_array.append(ifft_filt)

    EO = [[None] * norient for _ in range(nscale)]

    max_energy = None

    for o in range(norient):
        angl = o * np.pi / norient

        sum_e   = zero.copy()
        sum_o   = zero.copy()
        sum_an  = zero.copy()
        energy  = zero.copy()

        # Rebuild ifft filters per orientation (matches original code)
        ifft_filter_array_o = []
        for s in range(nscale):
            filt = log_gabor[s] * spread[o]
            ifft_filt = np.real(np.fft.ifft2(filt)) * np.sqrt(rows * cols)
            ifft_filter_array_o.append(ifft_filt)

        max_an = None
        EM_n = None

        for s in range(nscale):
            filt = log_gabor[s] * spread[o]
            eo = np.fft.ifft2(image_fft * filt)
            EO[s][o] = eo

            an = np.abs(eo)
            sum_an += an
            sum_e  += np.real(eo)
            sum_o  += np.imag(eo)

            if s == 0:
                EM_n  = np.sum(filt ** 2)
                max_an = an.copy()
            else:
                max_an = np.maximum(max_an, an)

        X_energy = np.sqrt(sum_e ** 2 + sum_o ** 2) + epsilon
        mean_e = sum_e / X_energy
        mean_o = sum_o / X_energy

        for s in range(nscale):
            E = np.real(EO[s][o])
            O = np.imag(EO[s][o])
            energy += E * mean_e + O * mean_o - np.abs(E * mean_o - O * mean_e)

        # Noise compensation
        median_e2n = np.median(np.abs(EO[0][o]) ** 2)
        mean_e2n   = -median_e2n / np.log(0.5)
        noise_power = mean_e2n / EM_n

        est_sum_an2  = sum(f ** 2 for f in ifft_filter_array_o)
        est_sum_ai_aj = zero.copy()
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                est_sum_ai_aj += ifft_filter_array_o[si] * ifft_filter_array_o[sj]

        sum_est_an2   = est_sum_an2.sum()
        sum_est_ai_aj = est_sum_ai_aj.sum()

        est_noise_energy2 = 2 * noise_power * sum_est_an2 + 4 * noise_power * sum_est_ai_aj
        tau = np.sqrt(est_noise_energy2 / 2)
        est_noise_energy = tau * np.sqrt(np.pi / 2)
        est_noise_sigma  = np.sqrt((2 - np.pi / 2) * tau ** 2)

        T = (est_noise_energy + k * est_noise_sigma) / 1.7
        energy = np.maximum(energy - T, 0.0)

        width  = sum_an / (max_an + epsilon) / nscale
        weight = 1.0 / (1.0 + np.exp((cut_off - width) * g))

        energy_this_orient = weight * energy
        total_sum_an      += sum_an
        total_energy      += energy_this_orient

        if o == 0:
            max_energy = energy_this_orient.copy()
        else:
            change      = energy_this_orient > max_energy
            orientation = (o) * change + orientation * (~change)
            max_energy  = np.maximum(max_energy, energy_this_orient)

        pc_o = weight * energy / (sum_an + epsilon)
        covx = pc_o * np.cos(angl)
        covy = pc_o * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    phase_cong = total_energy / (total_sum_an + epsilon)
    orientation = orientation * (180.0 / norient)

    # Principal moments of the covariance matrix → edge (M) and corner (m) strength
    covx2 /= norient / 2
    covy2 /= norient / 2
    covxy /= norient   # == 2*covxy / (norient/2) as in MATLAB
    denom     = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    sin2theta = covxy / denom
    cos2theta = (covx2 - covy2) / denom
    or_map    = np.arctan2(sin2theta, cos2theta) / 2
    or_map    = np.round(or_map * 180 / np.pi).astype(int)
    or_map    = np.where(or_map < 0, or_map + 180, or_map)

    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    return phase_cong, or_map, M, m


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from PIL import Image


    def load_gray(path):
        return np.array(Image.open(path).convert("L"))
    A = load_gray('test_img/MRI/25052.bmp')
    B = load_gray('test_img/PET/25052.bmp')
    F = load_gray('test_img/Fused/Fused_Gray.bmp')

    # A = np.array([
    #     [80, 20, 85],
    #     [75, 25, 78],
    #     [80, 22, 88]
    # ], dtype=np.float64)

    # B = np.array([
    #     [30, 110, 35],
    #     [28, 120, 32],
    #     [26, 115, 30]
    # ], dtype=np.float64)

    # F = np.array([
    #     [58, 70, 60],
    #     [55, 78, 57],
    #     [62, 75, 61]
    # ], dtype=np.float64)


    score = qp(A, B, F)
    print(f"QP metric: {score:.6f}")





