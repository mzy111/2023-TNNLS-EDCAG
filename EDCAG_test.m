clc; clear all; close all;
%% Data
DataName = 'PenDigits';
load([DataName,'_data.mat'])

%% Parameter
[n,d] = size(X);
c = length(unique(label));
m = 512;  % # anchors
k = 50;   % # neighbors of anchor graph

%% Anchor Generation
tic
[B,Anchor]=ULGEmzy(X,log2(m),k,1);
time1 = toc; % time of anchor generation + anchor graph construction

%% EDCAG
[labelnew,obj,~,~,converge,~,~,time2] = EDCAG(B,c);  % time2: time of discrete clustering

result = ClusteringMeasure_All(label,labelnew);
Result = [result(1:3) obj(1) max(obj) converge time1 time2 time1+time2 ];
% 1 ACC 2 NMI 3 ARI 4 Obj_init 5 Obj_final 6 converge 7 tAnchorG+tConstructB 8 tEDCAG 9 Time_sum