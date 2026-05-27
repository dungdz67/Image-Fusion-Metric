import numpy as np
from scipy.ndimage import convolve, uniform_filter
from scipy.signal import windows
from PIL import Image
from scipy.signal import convolve2d
import time

# ──────────────────────────────────────────────
# Structural Similarity (SSIM)
# ──────────────────────────────────────────────
# ---------------------------------------------------------------------------
# Gaussian kernel  (mirrors images.internal.createGaussianKernel)
# ---------------------------------------------------------------------------
 
def _create_gaussian_kernel(sigma, hsize):
    """
    Create a normalised 1-D Gaussian kernel (used for separable filtering).
 
    Parameters
    ----------
    sigma : float
    hsize : int
 
    Returns
    -------
    h : ndarray, shape (hsize,)
    """
    sigma  = float(sigma)
    hsize  = int(hsize)
    radius = (hsize - 1) / 2.0
    x      = np.arange(-radius, radius + 1)
    h      = np.exp(-0.5 * (x * x) / (sigma * sigma))
 
    # Suppress near-zero components (mirrors MATLAB)
    h[h < np.finfo(float).eps * h.max()] = 0.0
 
    s = h.sum()
    if s != 0:
        h = h / s
 
    return h
 
 
# ---------------------------------------------------------------------------
# imgaussfilt  (separable spatial path – sufficient for ssim)
# ---------------------------------------------------------------------------
def _scipy_pad_mode(padding):
    mapping = {'replicate': 'nearest', 'symmetric': 'reflect', 'circular': 'wrap'}
    if isinstance(padding, str):
        return mapping.get(padding, 'nearest')
    return 'constant'
 
 
def _imgaussfilt(A, sigma, filt_size, padding='replicate'):
    """
    2-D Gaussian filter – separable spatial implementation matching MATLAB.
 
    sigma     : scalar or 2-element [sigma_row, sigma_col]
    filt_size : scalar or 2-element [size_row, size_col]
    """
    sigma     = np.broadcast_to(np.atleast_1d(np.asarray(sigma,     dtype=float)), (2,)).copy()
    filt_size = np.broadcast_to(np.atleast_1d(np.asarray(filt_size, dtype=int)),   (2,)).copy()
 
    hcol = _create_gaussian_kernel(sigma[0], filt_size[0])  # column (vertical) kernel
    hrow = _create_gaussian_kernel(sigma[1], filt_size[1])  # row (horizontal) kernel
 
    pad_mode = _scipy_pad_mode(padding)
 
    # Separable convolution: apply hcol along axis-0, hrow along axis-1
    out = convolve(A.astype(float, copy=False), hcol[:, np.newaxis], mode=pad_mode)
    out = convolve(out,                          hrow[np.newaxis, :], mode=pad_mode)
    return out
 

# ---------------------------------------------------------------------------
# Core algorithm  (mirrors ssimalgo.m)
# ---------------------------------------------------------------------------
 
def _guarded_divide_and_exponent(num, den, C, exponent):
    if C > 0:
        component = num / den
    else:
        component = np.ones_like(num)
        nz = den != 0
        component[nz] = num[nz] / den[nz]
 
    if exponent != int(exponent):          # fractional exponent: clamp negatives
        component = np.maximum(component, 0.0)
 
    if exponent != 1:
        component = component ** exponent
 
    return component
 
 
def _ssimalgo(A, ref, gauss_fn, exponents, C, num_spatial_dims):
    """Direct translation of ssimalgo.m."""
 
    mux2 = gauss_fn(A)
    muy2 = gauss_fn(ref)
    muxy = mux2 * muy2
    mux2 = mux2 ** 2
    muy2 = muy2 ** 2
 
    sigmax2 = np.maximum(gauss_fn(A   ** 2) - mux2, 0.0)
    sigmay2 = np.maximum(gauss_fn(ref ** 2) - muy2, 0.0)
    sigmaxy = gauss_fn(A * ref) - muxy
 
    # Special case: equation 13 (Wang 2004)
    if C[2] == C[1] / 2 and np.array_equal(exponents, [1.0, 1.0, 1.0]):
        num = (2.0 * muxy + C[0]) * (2.0 * sigmaxy + C[1])
        den = (mux2 + muy2 + C[0]) * (sigmax2 + sigmay2 + C[1])
        if C[0] > 0 and C[1] > 0:
            ssimmap = num / den
        else:
            ssimmap = np.ones_like(A)
            nz = den != 0
            ssimmap[nz] = num[nz] / den[nz]
 
    else:
        # General case: equation 12
        ssimmap = np.ones_like(A) if exponents[0] == 0 else \
                  _guarded_divide_and_exponent(2.0 * muxy + C[0],
                                               mux2 + muy2 + C[0],
                                               C[0], exponents[0])
        sigmaxsigmay = None
        if exponents[1] > 0:
            sigmaxsigmay = np.sqrt(sigmax2 * sigmay2)
            ssimmap = ssimmap * _guarded_divide_and_exponent(
                2.0 * sigmaxsigmay + C[1], sigmax2 + sigmay2 + C[1], C[1], exponents[1])
 
        if exponents[2] > 0:
            if sigmaxsigmay is None:
                sigmaxsigmay = np.sqrt(sigmax2 * sigmay2)
            ssimmap = ssimmap * _guarded_divide_and_exponent(
                sigmaxy + C[2], sigmaxsigmay + C[2], C[2], exponents[2])
 
    # Mean over spatial dimensions (axes 0 and 1 for 2-D spatial)
    axis    = tuple(range(num_spatial_dims))
    ssimval = ssimmap.mean(axis=axis)
    return ssimval, ssimmap
 
# ---------------------------------------------------------------------------
# Dynamic-range helper
# ---------------------------------------------------------------------------
 
def _dynamic_range_from_dtype(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.max - info.min)
    return 1.0          # float32 / float64
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def ssim(A, ref,
         dynamic_range=None,
         regularization_constants=None,
         exponents=None,
         radius=1.5):
    """
    Structural Similarity Index (SSIM).
 
    Python port of MATLAB's offical SSIM ``ssim(A, ref, ...)`` (Image Processing Toolbox).
 
    Parameters
    ----------
    A, ref : array-like
        Images to compare.  Must have identical shape and dtype.
        Supported dtypes: uint8, uint16, int16, float32, float64.
        2-D (grayscale) or 3-D (rows × cols × channels).
 
    dynamic_range : float, optional
        Peak signal range.  Defaults to the theoretical range of the dtype
        (255 for uint8, 65535 for uint16, 1.0 for floats).
 
    regularization_constants : sequence of 3 floats, optional
        [C1, C2, C3].  If omitted, follows MATLAB default:
        [(0.01·DR)², (0.03·DR)², (0.03·DR)²/2].
 
    exponents : sequence of 3 floats, optional
        Exponents for luminance, contrast, and structure terms.
        Default [1, 1, 1].
 
    radius : float, optional
        Standard deviation of the Gaussian weighting window.  Default 1.5.
 
    Returns
    -------
    ssimval : float or ndarray
        Mean SSIM.  Scalar for 2-D input;
        1-D array (one per channel) for 3-D input.
    ssimmap : ndarray
        Local SSIM map, same shape as the input.
 
    Notes
    -----
    * int16 inputs are offset by ``intmin('int16')`` (−32768) before
      processing, exactly as MATLAB does.
    * Other integer types are cast to float64.
    * The Gaussian filter uses 'replicate' (nearest-neighbour) boundary
      padding, matching MATLAB's default.
    """
    A   = np.asarray(A, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
 
    if A.dtype != ref.dtype:
        raise TypeError("A and ref must have the same dtype "
                        f"(got {A.dtype} and {ref.dtype}).")
    if A.shape != ref.shape:
        raise ValueError("A and ref must have the same shape "
                         f"(got {A.shape} and {ref.shape}).")
    if A.ndim < 2 or A.ndim > 3:
        raise ValueError("Only 2-D and 3-D inputs are supported.")
 
    if exponents is None:
        exponents = np.array([1.0, 1.0, 1.0])
    else:
        exponents = np.asarray(exponents, dtype=float)
        if exponents.shape != (3,):
            raise ValueError("exponents must have exactly 3 elements.")
 
    if dynamic_range is None:
        dynamic_range = _dynamic_range_from_dtype(A.dtype)
    DR = float(dynamic_range)
    # print(dynamic_range)
    # --- dtype handling (mirrors ssimParseInputs.m) ---
    if A.dtype == np.int16:
        offset = float(np.iinfo(np.int16).min)   # -32768
        A   = A.astype(float) - offset
        ref = ref.astype(float) - offset
    elif np.issubdtype(A.dtype, np.integer):
        A   = A.astype(float)
        ref = ref.astype(float)
    else:
        A   = A.astype(float, copy=False)
        ref = ref.astype(float, copy=False)
 
    # --- regularisation constants ---
    if regularization_constants is None:
        C = np.array([(0.01 * DR) ** 2,
                      (0.03 * DR) ** 2,
                      (0.03 * DR) ** 2 / 2.0])
    else:
        C = np.asarray(regularization_constants, dtype=float)
        if C.shape != (3,):
            raise ValueError("regularization_constants must have exactly 3 elements.")
    # print(exponents)
    # print(C)
    # --- filter size (mirrors ssimParseInputs.m) ---
    filt_radius = int(np.ceil(radius * 3))   # 3 std-devs cover >99 % of area
    filt_size   = 2 * filt_radius + 1
 
    num_spatial_dims = 2   # rows and columns are always spatial
 
    # For 3-D input MATLAB processes slices independently via the 2-D gauss filter
    if A.ndim == 3:
        n_channels = A.shape[2]
 
        def gauss_fn_3d(X):
            out = np.empty_like(X)
            for c in range(n_channels):
                out[:, :, c] = _imgaussfilt(X[:, :, c], radius, filt_size)
            return out
        gauss_fn = gauss_fn_3d
    else:
        gauss_fn = lambda X: _imgaussfilt(X, radius, filt_size)
 
    ssimval, ssimmap = _ssimalgo(A, ref, gauss_fn, exponents, C, num_spatial_dims)
    return ssimval

def gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Create a 2D Gaussian window, normalised to sum to 1."""
    k = windows.gaussian(size, sigma)
    w = np.outer(k, k)
    return w / w.sum()


def ssim_index(
    img1: np.ndarray,
    img2: np.ndarray,
    K: tuple[float, float] = (0.01, 0.03),
    window: np.ndarray | None = None,
    L: int = 255,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Structural SIMilarity (SSIM) index between two images.

    Translated from the MATLAB implementation by Zhou Wang (2003).
    Reference:
        Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
        "Image quality assessment: From error measurement to structural
        similarity", IEEE Transactions on Image Processing, vol. 13, 2004.

    Parameters
    ----------
    img1 : np.ndarray
        First image (2-D grayscale, dtype float or uint8).
    img2 : np.ndarray
        Second image, same shape as img1.
    K : tuple[float, float]
        Stability constants (K1, K2). Default: (0.01, 0.03).
    window : np.ndarray or None
        Local weighting window. If None, an 11×11 Gaussian (σ=1.5) is used.
    L : int
        Dynamic range of the images. Default: 255.

    Returns
    -------
    mssim : float
        Mean SSIM index. Equal to 1.0 when img1 == img2.
    ssim_map : np.ndarray
        Per-pixel SSIM map (smaller than input by window size − 1).
    sigma1_sq : np.ndarray
        Local variance of img1.
    sigma2_sq : np.ndarray
        Local variance of img2.

    Raises
    ------
    ValueError
        If inputs are invalid (shape mismatch, image too small, bad K values).
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same shape, got {img1.shape} and {img2.shape}."
        )

    M, N = img1.shape[:2]

    if window is None:
        if M < 11 or N < 11:
            raise ValueError(
                "Images must be at least 11×11 pixels when using the default window."
            )
        window = gaussian_window(11, 1.5)

    H, W = window.shape
    if H * W < 4 or H > M or W > N:
        raise ValueError(
            "Window must have at least 4 elements and must not exceed image dimensions."
        )

    K1, K2 = K
    if K1 < 0 or K2 < 0:
        raise ValueError("K values must be non-negative.")

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = window / window.sum()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    def _filt(img: np.ndarray) -> np.ndarray:
        """2-D convolution in 'valid' mode (same as MATLAB filter2 + 'valid')."""
        return convolve(img, window, mode="constant", cval=0.0)[
            H // 2 : M - (H - 1 - H // 2),
            W // 2 : N - (W - 1 - W // 2),
        ]

    mu1 = _filt(img1)
    mu2 = _filt(img2)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filt(img1 * img1) - mu1_sq
    sigma2_sq = _filt(img2 * img2) - mu2_sq
    sigma12   = _filt(img1 * img2) - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = (
            (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
    else:
        numerator1   = 2 * mu1_mu2  + C1
        numerator2   = 2 * sigma12  + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones_like(mu1)

        mask_both = (denominator1 * denominator2) > 0
        ssim_map[mask_both] = (
            numerator1[mask_both] * numerator2[mask_both]
        ) / (denominator1[mask_both] * denominator2[mask_both])

        mask_one = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[mask_one] = numerator1[mask_one] / denominator1[mask_one]

    mssim = float(ssim_map.mean())

    return mssim, ssim_map, sigma1_sq, sigma2_sq


# ──────────────────────────────────────────────
# MS_SSIM  –  Multi-Scale SSIM for MEF
# ──────────────────────────────────────────────

def _extract_patches(arr, wsize):
    """
    arr   : H×W  or  H×W×N
    returns (H', W', wsize, wsize)  or  (H', W', wsize, wsize, N)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if arr.ndim == 2:
        return sliding_window_view(arr, (wsize, wsize))          # (H', W', w, w)
    else:
        H, W, N = arr.shape
        views = []
        for k in range(N):
            views.append(sliding_window_view(arr[:, :, k], (wsize, wsize)))
        return np.stack(views, axis=-1)                          # (H', W', w, w, N)


def _mef_ssim(seq, fi, wsize=11, K=0.03):
    """
    Vectorised single-scale MEF-SSIM.
    seq : H×W×N  stack of source images
    fi  : H×W    fused image
    """
    seq = seq.astype(np.float64)
    fi  = fi.astype(np.float64)

    H, W, N = seq.shape
    bd = wsize // 2
    C  = (K * 255) ** 2

    # ------------------------------------------------------------------ #
    # 1.  Per-pixel statistics (same as before, O(HWN) uniform filters)   #
    # ------------------------------------------------------------------ #
    mu    = uniform_filter(seq,      size=(wsize, wsize, 1))[bd:-bd, bd:-bd, :]   # (H', W', N)
    sigma = uniform_filter(seq ** 2, size=(wsize, wsize, 1))[bd:-bd, bd:-bd, :] - mu ** 2
    ed    = np.sqrt(np.maximum(wsize ** 2 * sigma, 0)) + 1e-3                     # (H', W', N)

    # ------------------------------------------------------------------ #
    # 2.  Gaussian window — kept as a flat vector                         #
    # ------------------------------------------------------------------ #
    ax   = np.arange(-5, 6)
    g    = np.exp(-(ax ** 2) / (2 * 1.5 ** 2))
    gwin = np.outer(g, g)
    gwin /= gwin.sum()
    gw   = gwin.ravel()                   # (w²,)

    # ------------------------------------------------------------------ #
    # 3.  Extract patches as one big batch                                 #
    #     seq_p : (H', W', w, w, N)  →  (B, w², N)   B = H'×W'          #
    #     fi_p  : (H', W', w, w)     →  (B, w²)                          #
    # ------------------------------------------------------------------ #
    seq_p = _extract_patches(seq, wsize)          # (H', W', w, w, N)
    fi_p  = _extract_patches(fi,  wsize)          # (H', W', w, w)

    Hp, Wp = seq_p.shape[:2]
    B  = Hp * Wp
    ww = wsize * wsize

    vecs = seq_p.reshape(B, ww, N)               # (B, w², N)
    fv   = fi_p.reshape(B, ww)                   # (B, w²)

    mu_b = mu.reshape(B, N)                       # (B, N)
    ed_b = ed.reshape(B, N)                       # (B, N)

    # ------------------------------------------------------------------ #
    # 4.  Structure-consistency weight  R → p → wk   (fully batched)      #
    # ------------------------------------------------------------------ #
    # vecs[:,k,:] - mu_b[:,k]   →  (B, w², N)
    centered = vecs - mu_b[:, np.newaxis, :]      # (B, w², N)

    # ||vecs[:,k] - mu_k||  for each source k  →  (B, N)
    denom = np.linalg.norm(centered, axis=1)      # (B, N)   (norm over w² pixels)

    # sum over sources first, then norm over pixels  →  (B,)
    sumvec   = centered.sum(axis=2)               # (B, w²)
    numerator = np.linalg.norm(
        sumvec - sumvec.mean(axis=1, keepdims=True), axis=1)  # (B,)

    R  = (numerator + 1e-10) / (denom.sum(axis=1) + 1e-10)   # (B,)
    R  = np.clip(R, 1e-10, 1 - 1e-10)

    p  = np.clip(np.tan(np.pi / 2 * R), 0, 10)               # (B,)

    # wk : edge-weighted, source-wise  →  (B, N)
    wk = (ed_b / wsize) ** p[:, np.newaxis]       # (B, N)
    wk = wk / (wk.sum(axis=1, keepdims=True) + 1e-10)

    maxEd = ed_b.max(axis=1)                      # (B,)

    # ------------------------------------------------------------------ #
    # 5.  Reference block  r = Σ_k  wk * (vecs_k - mu_k) / ed_k          #
    #     shape: (B, w²)                                                  #
    # ------------------------------------------------------------------ #
    rblock = (wk[:, np.newaxis, :] * centered
              / (ed_b[:, np.newaxis, :] + 1e-10)
             ).sum(axis=2)                         # (B, w²)

    nrm = np.linalg.norm(rblock, axis=1, keepdims=True)      # (B, 1)
    safe_nrm = np.where(nrm > 0, nrm, 1.0)          # replace 0 → 1 so division is always safe
    rblock = rblock / safe_nrm * maxEd[:, np.newaxis] # now no zero-division, ever                                         # (B, w²)

    # ------------------------------------------------------------------ #
    # 6.  Gaussian-weighted SSIM between rblock and fi patch              #
    # ------------------------------------------------------------------ #
    # gw : (w²,) — broadcast over batch
    mu1 = (gw * rblock).sum(axis=1)               # (B,)
    mu2 = (gw * fv).sum(axis=1)                   # (B,)

    rv_c = rblock - mu1[:, np.newaxis]            # (B, w²)
    fv_c = fv    - mu2[:, np.newaxis]             # (B, w²)

    s1  = (gw * rv_c ** 2).sum(axis=1)            # (B,)
    s2  = (gw * fv_c ** 2).sum(axis=1)            # (B,)
    s12 = (gw * rv_c * fv_c).sum(axis=1)          # (B,)

    qmap = (2 * s12 + C) / (s1 + s2 + C)         # (B,)

    return qmap.mean()

def _downsample(arr):
    """
    Box-filter (2×2 average) + stride-2 decimation.
    arr : H×W  or  H×W×N  — handled uniformly.
    """
    # Separable 2-tap box filter: blur then subsample
    # uniform_filter with size=2 is equivalent to the (2,2)/4 convolution
    if arr.ndim == 2:
        blurred = uniform_filter(arr, size=(2, 2), mode='mirror')
        return blurred[::2, ::2]
    else:
        # Apply the spatial filter to H and W axes only; leave N axis untouched
        blurred = uniform_filter(arr, size=(2, 2, 1), mode='mirror')
        return blurred[::2, ::2, :]


def ms_ssim(img_seq, fI, K=0.03, level=3):
    """
    Multi-scale MEF-SSIM

    img_seq : H×W×N  stack of source images
    fI      : H×W    fused image
    """
    weight = np.array([0.0448, 0.2856, 0.3001])[:level]
    weight = weight / weight.sum()

    img_seq = img_seq.astype(np.float64)
    fI      = fI.astype(np.float64)

    Q = np.empty(level)
    for l in range(level):
        Q[l] = _mef_ssim(img_seq, fI, K=K)
        if l < level - 1:
            img_seq = _downsample(img_seq)   # (H', W', N) — one call, no channel loop
            fI      = _downsample(fI)        # (H', W')

    return float(np.prod(Q ** weight))


# ──────────────────────────────────────────────
# FSIM  –  Feature SIMilarity
# ──────────────────────────────────────────────

def fsim(image_ref: np.ndarray, image_dis: np.ndarray):
    """
    Compute FSIM and FSIMc between a reference and a distorted image.

    Parameters
    ----------
    image_ref : np.ndarray
        Reference image.  Shape (H, W) for grayscale or (H, W, 3) for colour.
        Pixel values should be in [0, 255].
    image_dis : np.ndarray
        Distorted image with the same shape as *image_ref*.

    Returns
    -------
    FSIM : float
        Luminance-only similarity score in (0, 1].
    FSIMc : float
        Colour-aware similarity score in (0, 1].
        Equals FSIM for grayscale inputs.
    """
    rows, cols = image_ref.shape[:2]

    # Initialise chrominance channels (used only for colour images)
    I1 = np.ones((rows, cols), dtype=np.float64)
    I2 = np.ones((rows, cols), dtype=np.float64)
    Q1 = np.ones((rows, cols), dtype=np.float64)
    Q2 = np.ones((rows, cols), dtype=np.float64)

    if image_ref.ndim == 3:                         # colour images
        r1, g1, b1 = (image_ref[:, :, c].astype(np.float64) for c in range(3))
        r2, g2, b2 = (image_dis[:, :, c].astype(np.float64) for c in range(3))

        Y1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
        Y2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2

        I1 = 0.596 * r1 - 0.274 * g1 - 0.322 * b1
        I2 = 0.596 * r2 - 0.274 * g2 - 0.322 * b2

        Q1 = 0.211 * r1 - 0.523 * g1 + 0.312 * b1
        Q2 = 0.211 * r2 - 0.523 * g2 + 0.312 * b2
    else:                                           # grayscale images
        Y1 = image_ref.astype(np.float64)
        Y2 = image_dis.astype(np.float64)

    # ------------------------------------------------------------------
    # Downsample to keep the smaller dimension near 256 px
    # ------------------------------------------------------------------
    min_dim = min(rows, cols)
    F = max(1, round(min_dim / 256))

    if F > 1:
        ave_kernel = np.ones((F, F), dtype=np.float64) / (F * F)
        Y1  = _average_and_subsample(Y1,  ave_kernel, F, rows, cols)
        Y2  = _average_and_subsample(Y2,  ave_kernel, F, rows, cols)
        I1  = _average_and_subsample(I1,  ave_kernel, F, rows, cols)
        I2  = _average_and_subsample(I2,  ave_kernel, F, rows, cols)
        Q1  = _average_and_subsample(Q1,  ave_kernel, F, rows, cols)
        Q2  = _average_and_subsample(Q2,  ave_kernel, F, rows, cols)

    # ------------------------------------------------------------------
    # Phase congruency maps
    # ------------------------------------------------------------------
    PC1 = _phasecong2(Y1) #
    PC2 = _phasecong2(Y2)

    # ------------------------------------------------------------------
    # Gradient magnitude maps (Scharr-like kernel from the original)
    # ------------------------------------------------------------------
    dx = np.array([[ 3,  0, -3],
                   [10,  0,-10],
                   [ 3,  0, -3]], dtype=np.float64) / 16.0
    dy = np.array([[ 3, 10,  3],
                   [ 0,  0,  0],
                   [-3,-10, -3]], dtype=np.float64) / 16.0

    grad1 = _gradient_magnitude(Y1, dx, dy)
    grad2 = _gradient_magnitude(Y2, dx, dy)

    # ------------------------------------------------------------------
    # FSIM
    # ------------------------------------------------------------------
    T1 = 0.85   # phase congruency stability constant
    T2 = 160.0  # gradient stability constant

    PC_sim  = (2 * PC1 * PC2 + T1)  / (PC1**2 + PC2**2 + T1)
    Grd_sim = (2 * grad1 * grad2 + T2) / (grad1**2 + grad2**2 + T2)

    PCm = np.maximum(PC1, PC2)                      # weighting map

    sim_matrix = Grd_sim * PC_sim * PCm
    FSIM = sim_matrix.sum() / PCm.sum()

    # ------------------------------------------------------------------
    # FSIMc (adds chrominance terms)
    # ------------------------------------------------------------------
    T3, T4 = 200.0, 200.0
    I_sim = (2 * I1 * I2 + T3) / (I1**2 + I2**2 + T3)
    Q_sim = (2 * Q1 * Q2 + T4) / (Q1**2 + Q2**2 + T4)

    lam = 0.03
    # Use abs() before the fractional power (values near −1 can go complex);
    # the sign is preserved by multiplying back — same intent as the MATLAB
    # real(...^lambda) cast, which silently drops imaginary parts.
    chroma = np.sign(I_sim * Q_sim) * np.abs(I_sim * Q_sim) ** lam

    sim_matrix_c = Grd_sim * PC_sim * chroma * PCm
    FSIMc = sim_matrix_c.sum() / PCm.sum()

    return float(FSIM)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def convolution(x, k):
    """
    Simulate MATLAB's conv2 with 'same' output size by performing a full convolution
    and then cropping the result to match the input size.
    """
    full = convolve2d(x, k, mode='full')
    kh, kw = k.shape
    h, w = x.shape
    return full[kh//2:kh//2+h, kw//2:kw//2+w]

def _average_and_subsample(channel, kernel, F, rows, cols):
    """Box-filter then subsample by factor F (mirrors MATLAB conv2 + indexing)."""
    smoothed = convolution(channel, kernel)
    return smoothed[::F, ::F]


def _gradient_magnitude(im, dx, dy):
    Ix = convolution(im, dx)
    Iy = convolution(im, dy)
    return np.sqrt(Ix**2 + Iy**2)


def _lowpass_filter(size, cutoff, n):
    """
    Butterworth low-pass filter in the frequency domain.

    Parameters
    ----------
    size   : (rows, cols)
    cutoff : float  — cutoff frequency, 0 < cutoff <= 0.5
    n      : int    — filter order
    """
    rows, cols = size

    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x**2 + y**2)
    f = ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))
    return f


def _phasecong2(im):
    """
    Phase congruency map (PC_2 measure).

    Adapted from Peter Kovesi's MATLAB phasecong2.
    """
    # Parameters (kept identical to the original)
    nscale        = 4
    norient       = 4
    min_wave_len  = 6
    mult          = 2
    sigma_on_f    = 0.55
    d_theta_on_sigma = 1.2
    k             = 2.0
    epsilon       = 1e-4

    theta_sigma = np.pi / norient / d_theta_on_sigma

    rows, cols = im.shape
    imagefft = fft2(im)

    zero = np.zeros((rows, cols), dtype=np.float64)

    # Frequency-domain coordinate grids
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols // 2, cols // 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows // 2, rows // 2) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x**2 + y**2)
    theta  = np.arctan2(-y, x)

    radius = ifftshift(radius)
    theta  = ifftshift(theta)
    radius[0, 0] = 1.0          # avoid log(0)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Low-pass envelope (applied to all log-Gabor filters)
    lp = _lowpass_filter((rows, cols), 0.45, 15)

    # Radial (scale) filter components
    log_gabor = []
    for s in range(nscale):
        wavelength = min_wave_len * mult**s
        fo = 1.0 / wavelength
        lg = np.exp(-(np.log(radius / fo))**2 / (2 * np.log(sigma_on_f)**2))
        lg *= lp
        lg[0, 0] = 0.0
        log_gabor.append(lg)

    # Angular (orientation) filter components
    spread = []
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sin_theta * np.cos(angl) - cos_theta * np.sin(angl)
        dc = cos_theta * np.cos(angl) + sin_theta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread.append(np.exp(-dtheta**2 / (2 * theta_sigma**2)))

    energy_all = np.zeros((rows, cols), dtype=np.float64)
    an_all     = np.zeros((rows, cols), dtype=np.float64)

    for o in range(norient):
        sum_e  = zero.copy()
        sum_o  = zero.copy()
        sum_an = zero.copy()
        energy = zero.copy()

        ifft_filter_arr = []
        EM_n   = 0.0
        EO_1   = None           # EO at scale 0 for this orientation

        for s in range(nscale):
            filt = log_gabor[s] * spread[o]
            ifft_filt = np.real(ifft2(filt)) * np.sqrt(rows * cols)
            ifft_filter_arr.append(ifft_filt)

            EO = ifft2(imagefft * filt)

            An      = np.abs(EO)
            sum_an += An
            sum_e  += np.real(EO)
            sum_o  += np.imag(EO)

            if s == 0:
                EM_n = np.sum(filt**2)
                EO_1 = EO

        # Weighted mean phase vector
        X_energy = np.sqrt(sum_e**2 + sum_o**2) + epsilon
        mean_e   = sum_e / X_energy
        mean_o   = sum_o / X_energy

        # Energy (phase congruency × amplitude)
        for s in range(nscale):
            filt = log_gabor[s] * spread[o]
            EO   = ifft2(imagefft * filt)
            E    = np.real(EO)
            O    = np.imag(EO)
            energy += E * mean_e + O * mean_o - np.abs(E * mean_o - O * mean_e)

        # Noise threshold estimation
        median_e2n = np.median(np.abs(EO_1)**2)
        mean_e2n   = -median_e2n / np.log(0.5)
        noise_power = mean_e2n / EM_n

        est_sum_an2 = sum(f**2 for f in ifft_filter_arr)
        est_sum_ai_aj = zero.copy()
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                est_sum_ai_aj += ifft_filter_arr[si] * ifft_filter_arr[sj]

        est_noise_energy2 = (2 * noise_power * np.sum(est_sum_an2)
                             + 4 * noise_power * np.sum(est_sum_ai_aj))

        tau               = np.sqrt(est_noise_energy2 / 2)
        est_noise_energy  = tau * np.sqrt(np.pi / 2)
        est_noise_sigma   = np.sqrt((2 - np.pi / 2) * tau**2)

        T = (est_noise_energy + k * est_noise_sigma) / 1.7   # empirical rescaling

        energy = np.maximum(energy - T, 0)
        energy_all += energy
        an_all     += sum_an

    return energy_all / an_all

def piella_metrics(
    img1: np.ndarray,
    img2: np.ndarray,
    fuse: np.ndarray,
    sw: int,
) -> float:
    """
    Compute the Piella fusion quality index between two source images and a
    fused image.
 
    Reference:
        G. Piella and H. Heijmans, "A new quality metric for image fusion",
        IEEE ICIP 2003.
 
    Original MATLAB implementation by Z. Liu @ NRCC, 4 Oct 2003.
 
    Parameters
    ----------
    img1 : np.ndarray
        First source image (2-D grayscale).
    img2 : np.ndarray
        Second source image, same shape as img1.
    fuse : np.ndarray
        Fused image, same shape as img1.
    sw : int
        Metric selector:
            1 → Q   — basic fusion quality index
            2 → Qw  — weighted fusion quality index
            3 → Qe  — edge-dependent fusion quality index
 
    Returns
    -------
    float
        The selected fusion quality metric value.
 
    Raises
    ------
    ValueError
        If `sw` is not 1, 2, or 3.
    """
    if sw not in (1, 2, 3):
        raise ValueError(f"sw must be 1, 2, or 3; got {sw}.")
 
    def _compute_lambda(i1: np.ndarray, i2: np.ndarray):
        """
        Compute per-pixel lambda (local variance weight) and the two SSIM maps
        of the fused image against each source.
        """
        _, _, sigma1_sq, sigma2_sq = ssim_index(i1, i2)
 
        buffer = sigma1_sq + sigma2_sq
        # Avoid division by zero: if both variances are zero, set each to 0.5
        zero_mask = (buffer == 0).astype(np.float64) * 0.5
        sigma1_sq = sigma1_sq + zero_mask
        sigma2_sq = sigma2_sq + zero_mask
        buffer    = sigma1_sq + sigma2_sq
 
        lam = sigma1_sq / buffer
        return lam, sigma1_sq, sigma2_sq
 
    def _edge_magnitude(img: np.ndarray) -> np.ndarray:
        """Prewitt-like edge magnitude map."""
        flt_x = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]], dtype=np.float64)
        flt_y = np.array([[ 1,  1,  1],
                           [ 0,  0,  0],
                           [-1, -1, -1]], dtype=np.float64)
        img_f = img.astype(np.float64)
        gx = convolve(img_f, flt_x, mode="reflect")
        gy = convolve(img_f, flt_y, mode="reflect")
        return np.sqrt(gx ** 2 + gy ** 2)
 
    if sw in (1, 2):
        lam, sigma1_sq, sigma2_sq = _compute_lambda(img1, img2)
        _, ssim_map1 = ssim_index(fuse, img1)[:2]
        _, ssim_map2 = ssim_index(fuse, img2)[:2]
 
        Q_map = lam * ssim_map1 + (1 - lam) * ssim_map2
 
        if sw == 1:
            return float(Q_map.mean())
 
        # sw == 2 — spatially weighted by max local variance
        stack = np.stack([sigma1_sq, sigma2_sq], axis=-1)
        Cw    = stack.max(axis=-1)
        cw    = Cw / Cw.sum()
        return float((cw * Q_map).sum())
 
    # sw == 3 — edge-dependent
    fuse_F = _edge_magnitude(fuse)
    img1_F = _edge_magnitude(img1)
    img2_F = _edge_magnitude(img2)
 
    lam, sigma1_sq, sigma2_sq = _compute_lambda(img1_F, img2_F)
    _, ssim_map1 = ssim_index(fuse_F, img1_F)[:2]
    _, ssim_map2 = ssim_index(fuse_F, img2_F)[:2]
 
    stack = np.stack([sigma1_sq, sigma2_sq], axis=-1)
    Cw    = stack.max(axis=-1)
    cw    = Cw / Cw.sum()
    Qw    = float((cw * (lam * ssim_map1 + (1 - lam) * ssim_map2)).sum())
 
    alpha = 1
    Qe    = Qw * Qw ** alpha   # = Qw^(1 + alpha)
    return Qe

 


if __name__ == "__main__":
    import cv2
    def load_gray(path):
        return np.array(Image.open(path).convert("L"))
    import time
    # A = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png')
    # B = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/4010.png')
    # F = load_gray('data/Fused_results/SPECT-MRI/ASFE-Fusion/4010.png')

    N = 11
    
    A = np.random.randint(0, 255, (N, N))
    B= np.random.randint(0, 255, (N, N))
    F = np.random.randint(0, 255, (N, N))

    # print("\nStructural Similarity (SSIM): %.6f" % (0.5 * ssim(A, F) + 0.5 * ssim(B, F)))
    # print("Multi-Scale Structural Similarity (MS-SSIM): %.6f" % ms_ssim(np.stack([A, B], axis=2), F))
    print("Peilla metrics:")
    print("     Q:",piella_metrics(A, B, F, sw=1))   # basic
    print("     Qw:", piella_metrics(A, B, F, sw=2))   # weighted
    print("     Qe:", piella_metrics(A, B, F, sw=3))   # edge-dependent)