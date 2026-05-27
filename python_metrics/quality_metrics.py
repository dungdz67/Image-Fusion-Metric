import numpy as np

# ──────────────────────────────────────────────
# MLI_error  –  Mean Luminance Intensity Error
# ──────────────────────────────────────────────
def mli_error(img) -> float:
    img = img.astype(np.float64)
    return img.mean()

# ──────────────────────────────────────────────
# SD  –  Standard Deviation
# ──────────────────────────────────────────────
def sd(F: np.ndarray) -> float:
    """Contrast via pixel intensity standard deviation."""
    F = F.astype(np.float64)
    m, n = F.shape
    u = F.mean()
    return float(np.sqrt(((F - u) ** 2).sum() / (m * n)))


# ──────────────────────────────────────────────
# AG  –  Average Gradient
# ──────────────────────────────────────────────
def ag(img: np.ndarray) -> float:
    """Sharpness via average gradient magnitude. Using central difference"""
    img = img.astype(np.float64)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    g = []
    for k in range(img.shape[2]):
        band = img[:, :, k]
        dzdx, dzdy = np.gradient(band)
        s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
        r, c = band.shape
        g.append(s.sum() / ((r - 1) * (c - 1)))
    return float(np.mean(g))

def sharpness_ag(F: np.ndarray) -> float:
    """Sharpness via average gradient magnitude. Using forward difference"""
    I = F.astype(np.float64) / 255.0

    # Gradient kiểu 'intermediate'
    Gx = np.zeros_like(I)
    Gy = np.zeros_like(I)

    # dI/dx = I(x+1) - I(x)
    Gx[:, :-1] = I[:, 1:] - I[:, :-1]

    # dI/dy = I(y+1) - I(y)
    Gy[:-1, :] = I[1:, :] - I[:-1, :]

    # Magnitude
    S = np.sqrt(Gx**2 + Gy**2)

    # Average Gradient
    f = np.sum(S) / Gx.size

    return f

# ──────────────────────────────────────────────
# MSE  –  Mean Squared Error
# ──────────────────────────────────────────────
def mse(img, ref):
    img = img.astype(np.float32)
    ref = ref.astype(np.float32)
    return ((img - ref) ** 2).mean()

def mse_f(A: np.ndarray, B: np.ndarray, F: np.ndarray, max_value = 255.0) -> float:
    """Average squared pixel error between fused and source images."""
    A, B, F = A / max_value, B / max_value, F / max_value
    mse_af = mse(A, F)  # MSE between A and F
    mse_bf = mse(B, F)  # MSE between B and F
    return float(0.5 * mse_af + 0.5 * mse_bf)


# ──────────────────────────────────────────────
# PSNR  –  Peak Signal-to-Noise Ratio
# ──────────────────────────────────────────────
def psnr(img, ref, max_value = 1.0) -> float:
    """Signal-to-noise ratio in dB derived from MSE."""
    err = mse(img, ref)
    if err == 0:
        return float("inf")
    return float(20 * np.log10(max_value / np.sqrt(err)))

def psnr_f(A: np.ndarray, B: np.ndarray, F: np.ndarray, max_value = 255.0) -> float:
    """Average PSNR between fused and source images."""
    A, B, F = A / max_value, B / max_value, F / max_value
    psnr_af = psnr(A, F)  # PSNR between A and F
    psnr_bf = psnr(B, F)  # PSNR between B and F
    return float(0.5 * psnr_af + 0.5 * psnr_bf)

if __name__ == "__main__":
    # Define matrices
    I_1 = np.array([
        [80, 20, 85],
        [75, 25, 78],
        [80, 22, 88]
    ], dtype=np.float64)

    I_2 = np.array([
        [30, 110, 35],
        [28, 120, 32],
        [26, 115, 30]
    ], dtype=np.float64)

    I_F = np.array([
        [58, 70, 60],
        [55, 78, 57],
        [62, 75, 61]
    ], dtype=np.float64)

    ag = ag(I_F)
    sharpness_ag_value = sharpness_ag(I_F)
    mse_value = mse_f(I_1, I_2, I_F)
    psnr_value = psnr_f(I_1, I_2, I_F)

    print("\n=== Metrics ===")
    print(f"Average Gradient (AG): {ag}")
    print(f"Sharpness (AG): {sharpness_ag_value:.4f}")
    print(f"Mean Squared Error (MSE): {mse_value:.4f}")
    print(f"PSNR: {psnr_value:.2f} dB")