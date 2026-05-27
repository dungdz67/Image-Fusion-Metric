% Reference from https://github.com/thfylsty/Objective-evaluation-for-image-fusion

function [MCE] = OCE_metrics(imga,imgb,imgf)
%% Over All Cross Entropy-���彻����
% OCE�����ּ��㷽������ƽ�������غ;����������أ�ֵԽС˵���ں�ͼ���Դͼ������ȡ����ϢԽ�࣬�ں�Ч��Խ�á�
G1=double(imga);
G2=double(imgb);
G=double(imgf);
[m1,n1,p1]=size(G1);
[m2,n2,p2]=size(G2);
[m,n]=size(G);
m2=m1;
n2=n1;
X1=zeros(1,256);
X2=zeros(1,256);
X=zeros(1,256);

CEAF=0;
%ͳ����ͼ���Ҷȼ�����
for i=1:m1
    for j=1:n1
        X1(G1(i,j)+1)=X1(G1(i,j)+1)+1;
        X2(G2(i,j)+1)=X2(G2(i,j)+1)+1;
        X(G(i,j)+1)=X(G(i,j)+1)+1;
    end
end
%������ͼ���Ҷȼ����س��ֵĸ���
for k=1:256
    P1(k)=X1(k)/(m1*n1);
    P(k)=X(k)/(m*n);
    if((P1(k)~=0)&(P(k)~=0))
        CEAF=CEAF+P1(k)*log2(P1(k)/P(k));
    end
end
% sprintf('Դͼ��A �� �ں�ͼ�� F ֮��Ľ�����Ϊ : %.4f ',CEAF)

CEBF=0;
for k=1:256
    P2(k)=X2(k)/(m2*n2);
    P(k)=X(k)/(m*n);
    if((P2(k)~=0)&(P(k)~=0))
        CEBF=CEBF+P2(k)*log2(P2(k)/P(k));
    end
end
% sprintf('Դͼ��B �� �ں�ͼ�� F ֮��Ľ�����Ϊ : %.4f ',CEBF)

MCE=(CEAF+CEBF)/2;
% sprintf('Դͼ�����ں�ͼ��֮���ƽ��������Ϊ : %.4f ',MCE)

% RCE=sqrt((CEAF+CEBF)/2);
% sprintf('Դͼ�����ں�ͼ��֮��ľ�����������Ϊ : %.4f ',RCE)

% [COUNTSA,X]=imhist(imga);
% % PA=COUNTSA/(M*N);  %modified histogram,������ʷֲ�-calculation of the probability distributation
% for i=1:length(X)
%     PA(i)=COUNTSA(i)/prod(size(imga));  
% end  %�÷���ͬ��,���Ǽ���P(i)��ʾ�Ҷ�ֵΪi��������ĿNi �� ͼ���������� N ֮�ȣ��ɿ���ͼ��Ĺ�һ��ֱ��ͼ
% 
% [COUNTSB,Y]=imhist(imgb);
% for i=1:length(Y)
%     PB(i)=COUNTSB(i)/prod(size(imgb));  
% end 
% 
% [COUNTSF,F]=imhist(imgf);
% for i=1:length(F)
%     PF(i)=COUNTSF(i)/prod(size(imgf));  
% end 
% 
% CEAF=0;
% for j=1:length(X)
%     if((PA(j)~=0)&(PF(j)~=0))
%         CEAF=CEAF+PA(j)*(log2(PA/PF));
%     end
% end
% sprintf('Դͼ��A �� �ں�ͼ�� F ֮��Ľ�����Ϊ : %.4f ',CEAF)
% 
% CEBF=0;
% for j=1:length(Y)
%     if((PB(j)~=0)&(PF(j)~=0))
%         CEBF=CEBF+PB(j)*(log2(PB/PF));
%     end
% end
% sprintf('Դͼ��B �� �ں�ͼ�� F ֮��Ľ�����Ϊ : %.4f ',CEBF);
% 
% MCE=(CEAF+CEBF)/2;
% sprintf('Դͼ�����ں�ͼ��֮���ƽ��������Ϊ : %.4f ',MCE)