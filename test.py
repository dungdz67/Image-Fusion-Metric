import numpy as np
from scipy.signal import fftconvolve


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
    covxy /= norient / 2   # == 2*covxy / (norient/2) as in MATLAB

    denom     = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    sin2theta = covxy / denom
    cos2theta = (covx2 - covy2) / denom
    or_map    = np.arctan2(sin2theta, cos2theta) / 2
    or_map    = np.round(or_map * 180 / np.pi).astype(int)
    or_map    = np.where(or_map < 0, or_map + 180, or_map)

    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    return phase_cong, or_map, M, m


A = np.tile(np.arange(1, 17).reshape(16, 1), (1, 16))


print(phase_congruency(A))