import numpy as np
import math
from scipy.signal import convolve2d

EPS = 1e-12

# ──────────────────────────────────────────────
# QABF  –  Quality of Image Fusion (Edge-based)
# ──────────────────────────────────────────────
def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

def qabf(pA, pB, pF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5;
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    # if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
    # 如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

    strA = pA
    strB = pB
    strF = pF

    # 数组旋转180度
    def flip180(arr):
        return np.flip(arr)

    # 相当于matlab的Conv2
    def convolution_zero_pad(k, data):
        k = flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        img_new = convolve2d(data, k, mode='valid')
        return img_new

    def convolution(x, k):
        """
        Simulate MATLAB's conv2 with 'same' output size by performing a full convolution
        and then cropping the result to match the input size.
        """
        full = convolve2d(x, k, mode='full')
        kh, kw = k.shape
        h, w = x.shape
        return full[kh//2:kh//2+h, kw//2:kw//2+w]

    def getArray(img):
        SAx = convolution(img, h3)  
        SAy = convolution(img, h1)  
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        n, m = img.shape
        aA = np.zeros((n, m))
        zero_mask = SAx == 0
        aA[~zero_mask] = np.arctan(SAy[~zero_mask] / SAx[~zero_mask])
        aA[zero_mask] = np.pi / 2
        # for i in range(n):
        #     for j in range(m):
        #         if (SAx[i, j] == 0):
        #             aA[i, j] = math.pi / 2
        #         else:
        #             aA[i, j] = math.atan(SAy[i, j] / SAx[i, j])
        return gA, aA

    # 对strB和strF进行相同的操作
    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    # the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(aA, gA, aF, gF):
        mask = (gA > gF)
        GAF = np.where(mask, gF / (gA + EPS), np.where(gA == gF, gF, gA / (gF + EPS)))

        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)

        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))

        QAF = QgAF * QaAF
        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    output = nume / deno
    return output

def petrovic_metrics(I1, I2, f):
    """
    Compute full Petrovic fusion quality metrics.

    Parameters:
        f  : fused image (2D numpy array)
        I1 : source image 1 (2D numpy array)
        I2 : source image 2 (2D numpy array)

    Returns:
        QABF  : Total information transferred from source to fused image
        LABF  : Total loss of information
        NABF  : Modified fusion artifacts (B. K. Shreyamsha Kumar)
        NABF1 : Fusion artifacts (Petrovic original)
    """
    # Parameters
    Td = 2
    wt_min = 0.001
    P = 1
    Lg = 1.5
    Nrg = 0.9999
    kg = 19
    sigmag = 0.5
    Nra = 0.9995
    ka = 22
    sigmaa = 0.5

    xrcw = f.astype(np.float64)
    x1   = I1.astype(np.float64)
    x2   = I2.astype(np.float64)

    # Edge strength & orientation
    gvA, ghA = sobel_fn(x1)
    gA = np.sqrt(ghA**2 + gvA**2)

    gvB, ghB = sobel_fn(x2)
    gB = np.sqrt(ghB**2 + gvB**2)

    gvF, ghF = sobel_fn(xrcw)
    gF = np.sqrt(ghF**2 + gvF**2)

    p, q = xrcw.shape

    # Relative edge strength
    gA_safe = np.where(gA == 0, 1.0, gA)
    gF_safe = np.where(gF == 0, 1.0, gF)
    gB_safe = np.where(gB == 0, 1.0, gB)

    gAF = np.where(
        (gA == 0) | (gF == 0), 0.0,
        np.where(gA > gF, gF / gA_safe, gA / gF_safe)
    )
    gBF = np.where(
        (gB == 0) | (gF == 0), 0.0,
        np.where(gB > gF, gF / gB_safe, gB / gF_safe)
    )
    # Relative edge orientation
    aA = np.where((gvA == 0) & (ghA == 0), 0, np.arctan2(gvA, ghA))
    aB = np.where((gvB == 0) & (ghB == 0), 0, np.arctan2(gvB, ghB))
    aF = np.where((gvF == 0) & (ghF == 0), 0, np.arctan2(gvF, ghF))

    aAF = np.abs(np.abs(aA - aF) - np.pi / 2) * 2 / np.pi
    aBF = np.abs(np.abs(aB - aF) - np.pi / 2) * 2 / np.pi

    # Edge preservation coefficients
    QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
    QAF  = np.sqrt(QgAF * QaAF)

    QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
    QBF  = np.sqrt(QgBF * QaBF)

    # Weights
    wtA = np.where(gA >= Td, gA**Lg, wt_min)
    wtB = np.where(gB >= Td, gB**Lg, wt_min)
    wt_sum = np.sum(wtA + wtB)

    # QABF: total fusion performance
    QAF_wtsum = np.sum(QAF * wtA) / wt_sum
    QBF_wtsum = np.sum(QBF * wtB) / wt_sum
    QABF = QAF_wtsum + QBF_wtsum

    # LABF: fusion loss
    rr = ((gF <= gA) | (gF <= gB)).astype(np.float64)
    LABF = np.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

    # NABF1: fusion artifacts (Petrovic original)
    artifact_mask = (gF > gA) & (gF > gB)
    na1 = np.where(artifact_mask, 2 - QAF - QBF, 0.0)
    NABF1 = np.sum(na1 * (wtA + wtB)) / wt_sum

    # NABF: modified fusion artifacts (B. K. Shreyamsha Kumar)
    na = artifact_mask.astype(np.float64)
    NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

    return QABF, LABF, NABF, NABF1

# ──────────────────────────────────────────────
# SF  –  Spatial Frequency
# ──────────────────────────────────────────────
def sf(F: np.ndarray) -> float:
    """Overall activity level via row and column frequency."""
    F = F.astype(np.float64)
    rf = np.diff(F, axis=0)
    cf = np.diff(F, axis=1)
    rf1 = np.sqrt((rf ** 2).mean())
    cf1 = np.sqrt((cf ** 2).mean())
    return float(np.sqrt(rf1 ** 2 + cf1 ** 2))

if __name__ == "__main__":
    import cv2
    from PIL import Image
    from time import time
    def load_gray(path):
        return np.array(Image.open(path).convert("L"))
    # def load_gray(path):
    #     return np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    
    
    A = load_gray('test_img/MRI/25052.bmp')
    B = load_gray('test_img/PET/25052.bmp')
    F = load_gray('test_img/Fused/Fused_Gray.bmp')
    
    qabf_value = qabf(A, B, F)
    print(f"Quality of Image Fusion (QABF): {qabf_value:.4f}")
    
    qabf_value_full, labf_value, nabf_value, nabf1_value = petrovic_metrics(A, B, F)
    print(f"Petrovic Metrics:")
    print(f"QABF: {qabf_value_full:.4f}")
    print(f"LABF: {labf_value:.4f}")
    print(f"NABF: {nabf_value:.4f}")
    print(f"NABF1: {nabf1_value:.4f}")
