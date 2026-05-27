function QTE = QTE_metrics(im1, im2, fused, q)
% Calculate normalized Tsallis fusion quality index Q_TE
% im1   -- source image A
% im2   -- source image B
% fused -- fused image F
% q     -- Tsallis parameter (real, q ~= 1)
%
% QTE = (I^q(A,F) + I^q(B,F)) / (H^q(A) + H^q(B) - I^q(A,B))

    I_AF = tsallis(im1,   fused, q);
    I_BF = tsallis(im2,   fused, q);
    I_AB = tsallis(im1,   im2,   q);
    H_A  = tsallis_entropy(im1,  q);
    H_B  = tsallis_entropy(im2,  q);
    % disp(['I_AF = ', num2str(I_AF)]);
    % disp(['I_BF = ', num2str(I_BF)]);
    % disp(['I_AB = ', num2str(I_AB)]);
    % disp(['H_A  = ', num2str(H_A)]);
    % disp(['H_B  = ', num2str(H_B)]);
    denom = H_A + H_B + I_AB;
    if denom == 0
        QTE = 0;
    else
        QTE = (I_AF + I_BF) / denom;
    end
end

function H = tsallis_entropy(im, q)
% Marginal Tsallis entropy: H^q = (1 - sum_i p_i^q) / (1 - q)
    im = double(im);
    N  = 256;
    h  = histcounts(im(:), 0:N) / numel(im);  % probability vector
    p  = h(h > 0);
    H  = (1 - sum(p .^ q)) / (1 - q);
end

% Reference from https://github.com/thfylsty/Objective-evaluation-for-image-fusion
function RES=tsallis(im1,im2,q)

% function RES=tsallis(im1,im2,q)
% 
% This function is caculate Tsallis entropy for two input images.
% im1   -- input image one;
% im2   -- input image two;
% q     -- constant
%
% RES    -- Tsallis joint entropy;
%
%
% Note: The input images need to be in the range of 0-255. 
%
% Z. Liu @ NRCC [July 17, 2009]

im1=double(im1);
im2=double(im2);

[hang,lie]=size(im1);
count=hang*lie;
N=256;

%% caculate the joint histogram
h=zeros(N,N);

for i=1:hang
    for j=1:lie
        % in this case im1->x, im2->y
        h(im1(i,j)+1,im2(i,j)+1)=h(im1(i,j)+1,im2(i,j)+1)+1;
    end
end

%% marginal histogram

% this operation converts histogram to probability
h=h./sum(h(:));

im1_marg=sum(h);
im2_marg=sum(h');

result=0;
for i=1:N
    for j=1:N
        buff=im1_marg(i)*im2_marg(j);
        if buff~=0
            result=result+h(i,j).^q/(buff).^(q-1);
        end
    end
end

RES=(1-result)/(1-q);
end