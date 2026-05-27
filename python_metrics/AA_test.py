import cv2
from PIL import Image
import numpy as np

from structure_metrics import *
from img_metrics import *
from info_metrics import *
from quality_metrics import *
from structure_metrics import *
from visual_metrics import *    


def load_gray(path):
    return np.array(Image.open(path).convert("L"))
A = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png')
B = load_gray('data/AANLIB/MyDatasets/SPECT-MRI/test/SPECT/4010.png')
F = load_gray('data/Fused_results/SPECT-MRI/ASFE-Fusion/4010.png')

# A = np.array([
#     [80, 20, 85],
#     [75, 25, 78],
#     [80, 22, 88]
# ], dtype=np.uint8)

# B = np.array([
#     [30, 110, 35],
#     [28, 120, 32],
#     [26, 115, 30]
# ], dtype=np.uint8)

# F = np.array([
#     [58, 70, 60],
#     [55, 78, 57],
#     [62, 75, 61]
# ], dtype=np.uint8)


# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(A, cmap='gray')
# axes[0].set_title('Source Image A (IR)')
# axes[0].axis('off')
# axes[1].imshow(B, cmap='gray')
# axes[1].set_title('Source Image B (VI)')
# axes[1].axis('off')
# axes[2].imshow(F, cmap='gray')
# axes[2].set_title('Fused Image F')
# axes[2].axis('off')
# plt.tight_layout()
# plt.show()

print("Image shapes:", A.shape, "Image range:", A.min(), "-", A.max(), "-", A.dtype)

# =========================================================
# QUALITY METRICS
# =========================================================
print("MLI: %.6f" % mli_error(F))
print("Standard Deviation (SD): %.6f" % sd(F))
print("Average Gradient (AG): %.6f" % ag(F))
print("Mean Squared Error (MSE): %.6f" % mse_f(A, B, F))
print("Peak Signal-to-Noise Ratio (PSNR): %.6f" % psnr_f(A, B, F))

# =========================================================
# INFO METRICS
# =========================================================
print("\nEntropy (EN): %.6f" % en(F))
print("Mutual Information (MI): %.6f" % mi(A, B, F))
print("Nonlinear Correlation Information Entropy (NCIE): %.6f" % ncie(A, B, F))
print("Sum of the Correlations of Differences (SCD): %.6f" % scd(A, B, F))


print("Feature Mutual Information (FMI_pixel): %.6f" % fmi(A, B, F))
print("Feature Mutual Information (FMI_dct): %.6f" % fmi(A, B, F, feature="dct"))
print("Feature Mutual Information (FMI_wavelet): %.6f" % fmi(A, B, F, feature="wavelet"))
print("Feature Mutual Information (FMI_edge): %.6f" % fmi(A, B, F, feature="edge"))


# =========================================================
# IMAGE METRICS
# =========================================================
print("\nQABF_fast: %.6f" % (qabf(A, B, F)))
qabf_full, labf_value, nabf_value, nabf1_value = petrovic_metrics(A, B, F)
print(f"Petrovic Metrics:")
print(f"    QABF: {qabf_full:.4f}")
print(f"    LABF: {labf_value:.4f}")
print(f"    NABF: {nabf_value:.4f}")
print(f"    NABF1: {nabf1_value:.4f}")

print(f"Spatial Frequency: {sf(F):.4f}")

# =========================================================
# STRUCTURE METRICS
# =========================================================
print("\nStructural Similarity (SSIM): %.6f" % (0.5 * ssim(A, F) + 0.5 * ssim(B, F)))
print("Multi-Scale Structural Similarity (MS-SSIM): %.6f" % ms_ssim(np.stack([A, B], axis=2), F))
print("Peilla metrics:")
print("     Q:",piella_metrics(A, B, F, sw=1))   # basic
print("     Qw:", piella_metrics(A, B, F, sw=2))   # weighted
print("     Qe:", piella_metrics(A, B, F, sw=3))   # edge-dependent)
# =========================================================
# VISUAL METRICS
# =========================================================
print("\nVisual Information Fidelity (VIF): %.6f" % (vif(A, F) + vif(B, F)))
print("Visual Information Fidelity Fusion (VIFF): %.6f" % viff(A, B, F))