current = fileparts(mfilename('fullpath'));
addpath(genpath(current))
addpath(fullfile(current, 'VIF', 'VIF'))
clc;

img1 = imread('test_img/PET/25052.bmp');
img2 = imread('test_img/MRI/25052.bmp');
img_f = imread('test_img/Fused/Fused_Gray.bmp');

% % ========================
% % TEST MATRIX 3x3
% % ========================
% img1 = uint8([80  20  85;
%               75  25  78;
%               80  22  88]);

% img2 = uint8([30 110  35;
%               28 120  32;
%               26 115  30]);

% img_f = uint8([58  70  60;
%                55  78  57;
%                62  75  61]);

% ========================
% TEST MATRIX 5x5
% ========================

% img1 = uint8([85  20  88  22  90;
%               80  25  82  24  84;
%               78  23  80  21  79;
%               82  26  85  25  87;
%               88  22  90  23  92]);

% img2 = uint8([30 110  32 115  35;
%               28 120  30 118  33;
%               26 115  28 112  31;
%               29 122  31 120  34;
%               27 118  29 116  32]);

% img_f = uint8([58  70  60  72  62;
%                55  78  57  76  59;
%                52  72  54  70  56;
%                56  80  58  78  60;
%                57  75  59  74  61]);

% ========================
% PREPROCESS
% ========================
if size(img1,3)>2, img1 = rgb2gray(img1); end
if size(img2,3)>2, img2 = rgb2gray(img2); end
if size(img_f,3)>2, img_f = rgb2gray(img_f); end

[s1, s2] = size(img1);
grey_level = 256;

img1_int = img1;
img2_int = img2;
img_f_int = img_f;

img1_float = im2double(img1)*255.0;
img2_float = im2double(img2)*255.0;
img_f_float = im2double(img_f)*255.0;

imgSeq = cat(3, img1_float, img2_float);

fprintf('Image size: %dx%d\n', s1, s2);

% =========================================================
% 1. GLOBAL METRICS (NO sliding window) → ALWAYS SAFE
% =========================================================

% EN   = EN_metrics(img_f_int);
% OCE  = OCE_metrics(img1_float, img2_float, img_f_float);
% MI   = MI_metrics(img1_int, img2_int, img_f_int, grey_level, 0);
% NMI   = MI_metrics(img1_int, img2_int, img_f_int, grey_level, 1);
% MLI_error = MLI_metrics(img1_float, img2_float, img_f_float);
% PSNR = PSNR_metrics(img1_float, img2_float, img_f_float);
% MSE  = MSE_metrics(img1_float, img2_float, img_f_float);
% SF   = SF_metrics(img_f_float);
% SD   = SD_metrics(img_f_float);
% AG   = AG_metrics(img_f_float);
% CC   = CC_metrics(img1_float, img2_float, img_f_float);
% SCD  = SCD_metrics(img1_float, img2_float, img_f_float);
% EI   = EI_metrics(img_f_float);
% QTE   = QTE_metrics(img1_int, img2_int, img_f_int, 0.43137); % Nava constants
% rSFe = rSFe_metrics(img1_float, img2_float, img_f_float);
% % =========================================================
% % QUALITY METRICS
% % =========================================================
% fprintf('MLI Error: %f\n', MLI_error);
% fprintf('Standard Deviation (SD): %f\n', SD);
% fprintf('Average Gradient (AG): %f\n', AG);
% fprintf('Mean Squared Error (MSE): %f\n', MSE);
% fprintf('Peak Signal-to-Noise Ratio (PSNR): %f\n', PSNR);

% % =========================================================
% % INFO METRICS
% % =========================================================
% fprintf('Entropy (EN): %f\n', EN);
% fprintf('Mutual Information (MI): %f\n', MI);

FMI_pixel = FMI_metrics(img1_float, img2_float, img_f_float, 'none');
FMI_gradient   = FMI_metrics(img1_float, img2_float, img_f_float, 'gradient');
FMI_dct   = FMI_metrics(img1_float, img2_float, img_f_float, 'dct');
FMI_w     = FMI_metrics(img1_float, img2_float, img_f_float, 'wavelet');
FMI_edge  = FMI_metrics(img1_float, img2_float, img_f_float, 'edge');

% Keep only FMI variants that exist in your Python structure
fprintf('Feature Mutual Information (FMI_pixel): %.6f\n', FMI_pixel);
fprintf('Feature Mutual Information (FMI_gradient): %.6f\n', FMI_gradient);
fprintf('Feature Mutual Information (FMI_dct): %.6f\n', FMI_dct);
fprintf('Feature Mutual Information (FMI_w): %.6f\n', FMI_w);
fprintf('Feature Mutual Information (FMI_edge): %.6f\n', FMI_edge);

% fprintf('Sum of Correlations of Differences (SCD): %f\n', SCD);

% % =========================================================
% % IMAGE METRICS
% % =========================================================

% % ── IMAGE ─────────────────────────────────────────────────────
% QABF_fast = QABF_metrics(img1_float, img2_float, img_f_float);
% [QABF, LABF, NABF, NABF1] = Petrovic_metrics(img_f_float, img1_float, img2_float);
% SF   = SF_metrics(img_f_float);
% disp("Petrovic Metrics:")
% fprintf('\nQABF_fast: %f\n', QABF_fast);
% fprintf('QABF: %f\n', QABF);
% fprintf('LABF: %f\n', LABF);
% fprintf('NABF: %f\n', NABF);
% fprintf('NABF1: %f\n', NABF1);
% fprintf('Spatial Frequency: %f\n', SF);


% % =========================================================
% % SMALL-NEIGHBOR METRICS (compute first)
% % =========================================================
% try


%     SSIM = SSIM_metrics(img1_float, img2_float, img_f_float);
%     [MEF_SSIM, ~, ~] = MEF_SSIM_metrics(imgSeq, img_f_float);

%     VIF  = VIF_metrics(img1_float, img_f_float) + VIF_metrics(img2_float, img_f_float);
%     VIF_full = vifvec(img_f_float, img1_float);
%     VIFF = VIFF_metrics(img1_float, img2_float, img_f_float);

%     % =========================================================
%     % STRUCTURE METRICS
%     % =========================================================
%     fprintf('\nStructural Similarity (SSIM): %f\n', SSIM);
%     fprintf('Multi-Scale SSIM (MEF-SSIM): %f\n', MEF_SSIM);

%     % =========================================================
%     % VISUAL METRICS
%     % =========================================================
%     fprintf('Visual Information Fidelity (VIF): %f\n', VIF);
%     fprintf('Visual Information Fidelity (VIF) Full: %f\n', VIF_full);
%     fprintf('Visual Information Fidelity Fusion (VIFF): %f\n', VIFF);

% catch
%     fprintf('[WARNING] Small-neighbor metrics failed (border issue)\n');
% end