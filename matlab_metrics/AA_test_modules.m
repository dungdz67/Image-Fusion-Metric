clc;
stats = @(x, name) fprintf([ ...
    '%s shape=(%d,%d) min=%.4f max=%.4f mean=%.4f sum=%.4f\n'], ...
    name, size(x,1), size(x,2), ...
    min(x(:)), max(x(:)), mean(x(:)), sum(x(:).^2));
% img1 = imread('data/AANLIB/MyDatasets/SPECT-MRI/test/MRI/4010.png');

% img1 = uint8([80  20  85;
%               75  25  78;
%               80  22  88]);



% [cA,cH,cV,cD] = dwt2(img1,'dmey');

% disp(stats(cA, "cA"))
[Lo_D,Hi_D] = wfilters('dmey','d');

fprintf('Lo_D = [\n')
fprintf('%.18e,\n', Lo_D)
fprintf(']\n')

% [wtype,fname] = wavemngr('fields','dmey','type','file')
% edit(fname)