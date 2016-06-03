clear;

% import PIE pose image datasets
load('PIE_pose27.mat');
nClass = length(unique(gnd));
fea = NormalizeFea(fea);
fea = fea';

% data clustering in the original feature space
label = kmeans(fea,nClass);
label = bestMap(gnd,label);
MIhat = MutualInfo(gnd,label);
AC = length(find(gnd == label))/length(gnd);
disp(['Kmeans in the originall space normalized mutual information:',num2str(MIhat)]);
disp(['Kmeans in the original space accuracy:',num2str(AC)]);

% data clustering in the NMF reduced feature space
[U,V] = NMF(fea,nClass,100);
label = kmeans(V,nClass);
label = bestMap(gnd,label);
MIhat = MutualInfo(gnd,label);
AC = length(find(gnd == label))/length(gnd);
disp(['Kmeans in the NMF space normalized mutual information:',num2str(MIhat)]);
disp(['Kmeans in the NMF space accuracy:',num2str(AC)]);

clear;