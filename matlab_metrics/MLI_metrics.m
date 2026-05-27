function MLI_error = MLI_metrics(A, B, F)
% AverageIntensityMetrics:  Average Intensity Metrics for Image Fusion
%
% INPUT:
%   A - Source image A (grayscale, uint8)
%   B - Source image B (grayscale, uint8)
%   F - Fused image (grayscale, uint8)
%
% OUTPUT:
%   MI_F - Mean Intensity of the fused image
%   MI_ref - Reference Mean Intensity calculated from source images

    A = double(A);
    B = double(B);
    F = double(F);
    
    [M, N] = size(F);
    
    MLI_A = mean(A(:));
    MLI_B = mean(B(:));
    MLI_F = mean(F(:));
    
    MLI_ref = (MLI_A + MLI_B) / 2;
    MLI_error = abs(MLI_F - MLI_ref) / MLI_ref;
end