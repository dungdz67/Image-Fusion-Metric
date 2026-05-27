import numpy as np
from scipy.ndimage import convolve
from skimage import color
from scipy.signal import correlate2d


# ──────────────────────────────────────────────
# VIF  –  Visual Information Fidelity
# ──────────────────────────────────────────────
def vif(ref: np.ndarray, dist: np.ndarray) -> float:
    """
    Visual Information Fidelity (VIF) — direct Python port of vifp_mscale.m
    (Sheikh & Bovik, 2006).

    Parameters
    ----------
    ref  : np.ndarray  – Reference image  (2-D float or uint8)
    dist : np.ndarray  – Distorted image  (2-D float or uint8)

    Returns
    -------
    float  – VIF score  (1.0 = perfect fidelity, lower = more distortion)
    """
    ref  = ref.astype(np.float64)
    dist = dist.astype(np.float64)

    sigma_nsq = 2.0        # HVS noise variance — matches MATLAB constant
    EPS       = 1e-10

    num = 0.0
    den = 0.0

    for scale in range(1, 5):                      # scale = 1..4

        # ── 1. Scale-dependent Gaussian window (matches MATLAB exactly) ──────
        N   = 2 ** (4 - scale + 1) + 1            # 17, 9, 5, 3
        sig = N / 5.0

        k      = np.arange(N) - N // 2
        g1d    = np.exp(-k**2 / (2 * sig**2))
        g1d   /= g1d.sum()
        win    = np.outer(g1d, g1d)                # (N×N) Gaussian kernel

        # ── 2. Pre-filter + subsample for scales 2-4 (matches MATLAB) ────────
        #      MATLAB: filter2(win, img, 'valid')  then  img(1:2:end, 1:2:end)
        if scale > 1:
            ref  = correlate2d(ref,  win, mode='valid')[::2, ::2]
            dist = correlate2d(dist, win, mode='valid')[::2, ::2]

        # ── 3. Local statistics via 'valid' convolution ───────────────────────
        mu1     = correlate2d(ref, win, mode='valid')
        mu2     = correlate2d(dist,win, mode='valid')

        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = correlate2d(ref  * ref,  win, mode='valid') - mu1_sq
        sigma2_sq = correlate2d(dist * dist, win, mode='valid') - mu2_sq
        sigma12   = correlate2d(ref  * dist, win, mode='valid') - mu1_mu2

        # Clamp negative variances (numerical noise)
        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)

        # ── 4. Distortion-channel gain g and residual noise sv_sq ─────────────
        g     = sigma12 / (sigma1_sq + EPS)
        sv_sq = sigma2_sq - g * sigma12

        # ── 5. Edge-case masking (mirrors MATLAB conditionals exactly) ─────────
        # Where reference variance is negligible → no signal to compare
        g    [sigma1_sq < EPS] = 0
        sv_sq[sigma1_sq < EPS] = sigma2_sq[sigma1_sq < EPS]
        sigma1_sq[sigma1_sq < EPS] = 0

        # Where distorted variance is negligible → no information transferred
        g    [sigma2_sq < EPS] = 0
        sv_sq[sigma2_sq < EPS] = 0

        # Negative gain is non-physical → clamp to zero
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g    [g < 0] = 0

        # Residual noise floor
        sv_sq[sv_sq <= EPS] = EPS

        # ── 6. VIF information ratio (log10, matches MATLAB) ──────────────────
        num += np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    return float(num / (den + EPS))

# ──────────────────────────────────────────────
# VIFF  –  Visual Information Fidelity Fusion
# ──────────────────────────────────────────────

def viff(im1: np.ndarray, im2: np.ndarray, imf: np.ndarray) -> float:
    """
    Compute the VIFF (Visual Information Fidelity for Fusion) metric.

    Parameters
    ----------
    im1 : np.ndarray
        Source image 1. Shape (H, W) or (H, W, 3), dtype uint8 or float.
    im2 : np.ndarray
        Source image 2. Same shape as im1.
    imf : np.ndarray
        Fused image. Same shape as im1.

    Returns
    -------
    float
        VIFF fusion quality score (higher = better).
    """
    # ---- visual noise constant (matches MATLAB sq = 0.005 * 255^2) --------
    sq = 0.005 * 255.0 * 255.0

    # ---- multi-scale weights (p vector from MATLAB) -----------------------
    p = np.array([1.0, 0.0, 0.15, 1.0]) / 2.15

    # ---- colour space: keep only luminance channel ------------------------
    ix1 = _luminance(im1)
    ix2 = _luminance(im2)
    ixf = _luminance(imf)

    # ---- compute per-scale VID / VIND / G for each source vs fused -------
    t1_n, t1_d, t1_g = _com_vid_vind_g(ix1, ixf, sq)
    t2_n, t2_d, t2_g = _com_vid_vind_g(ix2, ixf, sq)

    # ---- aggregate across 4 scales ----------------------------------------
    f_scales = np.zeros(4)
    c = 1e-7  # error comparison epsilon

    for i in range(4):
        m_z1 = t1_n[i]
        m_z2 = t2_n[i]
        m_m1 = t1_d[i]
        m_m2 = t2_d[i]
        m_g1 = t1_g[i]
        m_g2 = t2_g[i]

        # Select the source with the *smaller* local gradient magnitude
        # (weaker source provides more complementary information)
        use_src1 = m_g1 < m_g2

        m_z12 = np.where(use_src1, m_z1, m_z2)
        m_m12 = np.where(use_src1, m_m1, m_m2)

        vid  = np.sum(m_z12 + c)
        vind = np.sum(m_m12 + c)

        f_scales[i] = vid / vind

    return float(np.dot(f_scales, p))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _luminance(img: np.ndarray) -> np.ndarray:
    """
    Return the luminance (L*) channel as a float64 array.

    For RGB inputs, converts sRGB → CIE L*a*b* and extracts L*.
    For grayscale inputs, returns a float64 copy as-is.

    MATLAB equivalent
    -----------------
    cform = makecform('srgb2lab');
    T = applycform(img, cform);
    L = T(:,:,1);
    """
    img = np.asarray(img, dtype=np.float64)

    if img.ndim == 3 and img.shape[2] == 3:
        # skimage.color.rgb2lab expects float in [0, 1] or uint8
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = color.rgb2lab(img_u8)          # → L* in [0, 100]
        return lab[:, :, 0]                  # keep only L*

    # Already grayscale
    return img.squeeze()


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Build a 2-D Gaussian kernel of given *size* and *sigma*.

    MATLAB equivalent: fspecial('gaussian', N, N/5)
    """
    ax = np.arange(size) - size // 2
    g1d = np.exp(-ax**2 / (2.0 * sigma**2))
    g2d = np.outer(g1d, g1d)
    return g2d / g2d.sum()


def _filter2_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D correlation with 'valid' boundary (no edge padding).

    MATLAB equivalent: filter2(win, img, 'valid')
    SciPy's convolve uses correlation when the kernel is symmetric,
    which Gaussian kernels are. We crop the border manually to replicate
    MATLAB's 'valid' output size: (H - kH + 1) × (W - kW + 1).
    """
    full = convolve(img, kernel, mode='reflect')
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    return full[pad_h: img.shape[0] - pad_h,
                pad_w: img.shape[1] - pad_w]


def _com_vid_vind_g(
    ref: np.ndarray,
    dist: np.ndarray,
    sq: float,
) -> tuple[list, list, list]:
    """
    Compute per-scale VID, VIND, and scalar gain (g) maps.

    Mirrors the MATLAB function ComVidVindG.

    Based on:
      H.R. Sheikh and A.C. Bovik, "Image information and visual quality,"
      IEEE Trans. Image Processing 15(2), pp. 430–444, 2006.

    Parameters
    ----------
    ref  : float64 luminance of a source image
    dist : float64 luminance of the fused image
    sq   : visual noise variance (σ_n²)

    Returns
    -------
    num_list : list of 4 arrays — VID maps per scale
    den_list : list of 4 arrays — VIND maps per scale
    g_list   : list of 4 arrays — scalar gain (g) maps per scale
    """
    sigma_nsq = sq
    r = ref.copy()
    d = dist.copy()

    num_list = []
    den_list = []
    g_list   = []

    for scale in range(1, 5):               # scales 1 … 4
        # ---- build Gaussian window for this scale ------------------------
        n   = 2 ** (4 - scale + 1) + 1     # N = 17, 9, 5, 3  for scales 1–4
        win = _gaussian_kernel(n, n / 5.0)

        # ---- downsample from scale 2 onward (skip at scale 1) -----------
        if scale > 1:
            r = _filter2_valid(r, win)
            d = _filter2_valid(d, win)
            r = r[::2, ::2]
            d = d[::2, ::2]

        # ---- local statistics -------------------------------------------
        mu1    = _filter2_valid(r, win)
        mu2    = _filter2_valid(d, win)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1mu2 = mu1 * mu2

        sigma1_sq = _filter2_valid(r * r, win) - mu1_sq
        sigma2_sq = _filter2_valid(d * d, win) - mu2_sq
        sigma12   = _filter2_valid(r * d, win) - mu1mu2

        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        # ---- scalar gain g (slope of the best linear estimator) ---------
        eps = 1e-10
        g     = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        # -- enforce non-negative constraints (mirrors MATLAB's if-blocks) -
        g[sigma1_sq < eps]     = 0.0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0.0

        g[sigma2_sq < eps]     = 0.0
        sv_sq[sigma2_sq < eps] = 0.0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0]     = 0.0

        sv_sq = np.maximum(sv_sq, eps)

        # ---- information measures ---------------------------------------
        # VID  = mutual information between source and fused (with distortion)
        # VIND = mutual information between source and ideal observer
        vid  = np.log10(1.0 + g**2 * sigma1_sq / (sv_sq + sigma_nsq))
        vind = np.log10(1.0 + sigma1_sq / sigma_nsq)

        num_list.append(vid)
        den_list.append(vind)
        g_list.append(g)

    return num_list, den_list, g_list

if __name__ == "__main__":
    from PIL import Image
    def load_gray(path):
        return np.array(Image.open(path).convert("L"))
    # A = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png')
    B = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/4010.png')
    F = load_gray('data/Fused_results/SPECT-MRI/ASFE-Fusion/4010.png')
    
    vif_val = vif(B, F)
    
    print(vif_val)
