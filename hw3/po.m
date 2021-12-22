path = imageDatastore('C:\Users\Mahesh\Desktop\digital image processing\HW3\venv\camera_man.tiff');
img = read(path);
%imshow(img);
db = im2double(img);
[m,n] = size(db );
p=2*m;
q= 2*n;
disp([p,q]);
con= zeros(p,q);

% STEP 2
B = padarray(b,[p/2,q/2],0,'post');
imshow(B);
imsave();

% STEP 3
for i = 1:p
    for j = 1:q
        con(i,j) = B(i,j).*(-1).^(i + j);
    end
end

imshow(con);
imsave();

% STEP 4
dft_2d = fft2(con);
imshow(dft_2d);
imsave();

% STEP 5

H=zeros(p,q);
x1= p/2;
y1= q/2;
si= 0.1;
for i=1:p
    for j=1:q
        H(i,j)=exp(-((i-x1).^2 + (j-y1).^2)/ 2*si ^2);
    end
end

imshow(H)
imsave();

% STEP 6

g= zeros(p,q);
g= dft_2d .* H;

imshow(g);
imsave();


% STEP 7

inv_dft = ifft2(g);

act_img= zeros(p,q);
for i = 1:p
    for j = 1:q
        act_img(i,j) = inv_dft(i,j).*(-1).^(i + j);
    end
end

imshow(act_img);
imsave();

% STEP 8

final_img = zeros(m,n);
for i = 1:m
    for j = 1:n
        final_img(i,j) = act_img(i,j);
    end
end

imshow(final_img);
imsave();
