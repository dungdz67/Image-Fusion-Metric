% Reference from https://github.com/thfylsty/Objective-evaluation-for-image-fusion

function res = MI_metrics(grey_matrixA, grey_matrixB, grey_matrixF, grey_level, sw)
% Function to combine many types of mutual information metrics
% Inputs:
%   grey_matrixA, grey_matrixB : source images (range 0-255)
%   grey_matrixF               : fused image   (range 0-255)
%   grey_level                 : grayscale levels (default: 256)
%   sw = 0 : Standard MI (Shannon)
%   sw = 1 : Normalized MI (Hossny)

if nargin < 5, sw = 0;        end
if nargin < 4, grey_level = 256; end

switch sw
    case 0
        % Standard MI (Shannon)
        [MI_FA, ~, ~, ~] = mutual_info(grey_matrixF, grey_matrixA);
        [MI_FB, ~, ~, ~] = mutual_info(grey_matrixF, grey_matrixB);
        res = MI_FA + MI_FB;

    case 1
        % Normalized MI (Hossny)
        [MI_FA, ~, H_F1, H_A] = mutual_info(grey_matrixF, grey_matrixA);
        [MI_FB, ~, H_F2, H_B] = mutual_info(grey_matrixF, grey_matrixB);
        res = 2 * (MI_FA / (H_F1 + H_A) + MI_FB / (H_F2 + H_B));

    otherwise
        error('Invalid sw value. Use 0 (Shannon MI) or 1 (Normalized MI).');
end
end


function [MI, H_xy, H_x, H_y] = mutual_info(im1, im2)

% function [MI,H_xy,H_x,H_y]=mutual_info(im1,im2)
% 
% This function is caculate the mutual information of two input images.
% im1   -- input image one;
% im2   -- input image two;
%
% MI    -- mutual information;
%
%
% Note: The input images need to be in the range of 0-255. (see function:
% normalize1.m)
%
% Z. Liu @ NRCC [July 17, 2009]

im1 = double(im1);
im2 = double(im2);

[hang, lie] = size(im1);
N = 256;

% Joint histogram
h = zeros(N, N);
for i = 1:hang
    for j = 1:lie
        h(im1(i,j)+1, im2(i,j)+1) = h(im1(i,j)+1, im2(i,j)+1) + 1;
    end
end

% Normalize to joint probability
h = h ./ sum(h(:));

% Marginal probabilities
im1_marg = sum(h);    % p(x)
im2_marg = sum(h');   % p(y)

% Shannon entropies
H_x  = -sum(im1_marg .* log2(im1_marg + (im1_marg == 0)));
H_y  = -sum(im2_marg .* log2(im2_marg + (im2_marg == 0)));
H_xy = -sum(sum(h .* log2(h + (h == 0))));

MI = H_x + H_y - H_xy;
end
