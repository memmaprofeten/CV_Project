%% Load
clear; clc;

% Gets a list of all traffic signs and index of image that belongs to a
% certain class
info = load('gt.txt');

% Class to load (1,2,3,4 etc)
classes = find(info(:,6)==5);

imageIDs = zeros(size(classes));

for i = 1:length(classes)
    imageIDs(i) = info(classes(i),1) 
end


%%

clf;

% This number is what image to load
ImageToLoad = 425;

dir1 = dir('*.ppm') ;
im = imread(dir1(ImageToLoad+1).name); 
imD = imread(dir1(31).name); %Reference image
im2 = imhistmatch(im,imD,256);


%
imLogic = createMask5(im2);

se = strel('disk',1);

imLogic = imopen(imLogic,se);
imLogic = imclose(imLogic,se);
imLogic = bwareaopen(imLogic,80);
imLogic = edge(imLogic,'Canny');


figure(1)
subplot(2,2,1)
imshow(im)
subplot(2,2,2)
imshow(im2);
subplot(2,2,3)
imshow(imLogic)


subplot(2,2,4)
figure(6)
imshow(im)
% Find circles
[centers,radii] = imfindcircles(imLogic,[15 100]);
viscircles(centers,radii,'color','b');

%% ROI
   

imROI = imcrop(im,[round(centers(1))-radii(1)-5 round(centers(2))-radii(1)-5 2*radii(1)+10 2*radii(1)+10]);
figure(4)
imshow(imROI)