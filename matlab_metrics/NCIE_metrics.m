% Reference from https://github.com/thfylsty/Objective-evaluation-for-image-fusion

function res=NCIE_metrics(im1,im2,fim)

% function res=metricWang(im1,im2,fim) -> NCIE Metric
%
% This function implements Wang's algorithms for fusion metric.
% im1, im2 -- input images;
% fim      -- fused image;
% res      -- metric value;
%
% IMPORTANT: The size of the images need to be 2X. 
% See also: NCC.m, mutual_info.m, evalu_fusion.m
%
% Z. Liu [July 2009]
%

% Ref: Performance evaluation of image fusion techniques, Chapter 19, pp.469-492, 
% in Image Fusion:  Algorithms and Applications, edited by Tania Stathaki
% by Qiang Wang
%

%% pre-processing
im1=normalize1(im1);
im2=normalize1(im2);
fim=normalize1(fim);

[hang,lie]=size(im1);
b=256;
K=3;

%% Call mutual_info.m
% two inputs

NCCxy=NCC(im1,im2);


% one input and fused image
NCCxf=NCC(im1,fim);


% another input and fused image
NCCyf=NCC(im2,fim);


%% get the correlation matrix and eigenvalue 

R=[ 1 NCCxy NCCxf; NCCxy 1 NCCyf; NCCxf NCCyf 1];
r=eig(R);

%% HR

HR=sum(r.*log2(r./K)/K);
HR=-HR/log2(b);

%% NCIE

NCIE=1-HR;

res=NCIE;

end


function res=NCC(im1,im2)

% function res=NCC(im1,im2)
% 
% This function is caculate the mutual information of two input images.
% im1   -- input image one;
% im2   -- input image two;
%
% res    -- NNC (nonlinear correlation coefficient
%
%
% Note: 1) The input images need to be in the range of 0-255. (see function:
% normalize1.m); 2) This function is similar to mutual information but they
% are different.
%
% Z. Liu @ NRCC [July 17, 2009]

im1=double(im1);
im2=double(im2);

[hang,lie]=size(im1);
count=hang*lie;
N=256;
b=256;

%% caculate the joint histogram
h=zeros(N,N);

for i=1:hang
    for j=1:lie
        % in this case im1->x (row), im2->y (column)
        h(im1(i,j)+1,im2(i,j)+1)=h(im1(i,j)+1,im2(i,j)+1)+1;
    end
end

%% marginal histogram

% this operation converts histogram to probability
%h=h./count;
h=h./sum(h(:));

im1_marg=sum(h);    % sum each column for im1
im2_marg=sum(h');   % sum each row for im2


%for i=1:N
%    if (im1_marg(i)>eps)
%        % entropy for image1
%        Hx=Hx+(-im1_marg(i)*(log2(im1_marg(i))));
%    end
%    if (im2_marg(i)>eps)
%        % entropy for image2
%        Hy=Hy+(-im2_marg(i)*(log2(im2_marg(i))));
%    end
%end

H_x=-sum(im1_marg.*log2(im1_marg+(im1_marg==0)));
H_y=-sum(im2_marg.*log2(im2_marg+(im2_marg==0)));


% joint entropy

%H_xy=0;

%for i=1:N
%    for j=1:N
%        if (h(i,j)>eps)
%            H_xy=H_xy+h(i,j)*log2(h(i,j));
%        end
%    end
%end

H_xy=-sum(sum(h.*log2(h+(h==0))));
H_xy=H_xy/log2(b);


%H_xy=-sum(sum(h.*(log2(h+(h==0)))));

H_x=H_x/log2(b);
H_y=H_y/log2(b);

% NCC
res=H_x+H_y-H_xy;
end


function RES=normalize1(data)

    % function RES=normalize1(data)
    %
    % This function is to NORMALIZE the data. 
    % The data will be in the interval 0-255 (gray level) and pixel value has
    % been rounded to an integer.
    % 
    % See also: normalize.m 
    %
    % Z. Liu @NRCC (Aug 24, 2009)

    data=double(data);
    da=max(data(:));
    xiao=min(data(:));
    if (da==0 & xiao==0)
        RES=data;
    else
        newdata=(data-xiao)/(da-xiao);
        RES=round(newdata*255);
    end
end