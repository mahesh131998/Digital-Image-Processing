%Q1)
%a)
BW = imread('apple.gif');
BW1 = imbinarize(BW);
imshow(BW1)
BW2=bwmorph(BW1,'skel',inf);
montage({BW,BW2},'BackgroundColor','blue','BorderSize',5)

BW3 = bwmorph(BW1,'thin',inf);
montage({BW,BW3},'BackgroundColor','blue','BorderSize',5)
imwrite(BW2, 'apple_skel.jpg');
imwrite(BW3, 'apple_thin.jpg');

 
%b)
BW4 = imread('cup.gif');
BW5 = imbinarize(BW4);
imshow(BW5)
BW6 = bwmorph(BW5,'skel',inf);
montage({BW4,BW6},'BackgroundColor','blue','BorderSize',5)

BW7 = bwmorph(BW5,'thin',inf);
montage({BW4,BW7},'BackgroundColor','blue','BorderSize',5)
imwrite(BW6, 'cup_skel.jpg');
imwrite(BW7, 'cup_thin.jpg');

%c)
BW8 = imread('tree.gif');
BW9 = imbinarize(BW8);
imshow(BW9)
BW10 = bwmorph(BW9,'skel',inf);
montage({BW8,BW10},'BackgroundColor','blue','BorderSize',5)

BW11 = bwmorph(BW9,'thin',inf);
montage({BW8,BW11},'BackgroundColor','blue','BorderSize',5)
imwrite(BW10, 'tree_skel.jpg');
imwrite(BW11, 'tree_thin.jpg');



% % Q2)
BW12 = imread('stars.png');
BW13 = imbinarize(BW12);
BW14=bwmorph(BW13, 'shrink', Inf); %shrink
[x1,y1,z1] = size(BW14);
 White_pix=0;
 Floc=0;
 for j=1:(x1)-1
    for i=1:(y1)-1
        if BW14(j,i)==1
            White_pix=White_pix+1;
        end
    end
 end

disp(" total number of stars are");
disp(White_pix);



