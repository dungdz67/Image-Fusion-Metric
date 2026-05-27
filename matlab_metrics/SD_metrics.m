% Reference from https://github.com/Linfeng-Tang/Image-Fusion General Metrics

function SD = SD_metrics(F)
[m,n]=size(F);
u=mean(mean(F));
SD=sqrt(sum(sum((F-u).^2))/(m*n));
end
