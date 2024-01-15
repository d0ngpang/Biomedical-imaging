clear all;
clc;
close all;

%% 1. Boundary detection by the lookup table(LUT)
k = 0:8; w = 2.^ k; w = reshape(w, 3, 3);
bdy1 = @ (x) (x(5) == 1) & (x(2)*x(4)*x(6)*x(8)) == 0;
bdy2 = @ (x) (x(5) == 1) & (x(1)*x(2)*x(3)*x(4)*x(6)*x(7)*x(8)*x(9)) == 0;
lut1 = makelut(bdy1, 3);
lut2 = makelut(bdy2, 3);

img = imread('lung.tif');

imw_4 = applylut(img, lut1);
imw_8 = applylut(img, lut2);
imw_df = imw_8 - imw_4;

figure, 
subplot(2, 3, 1), plot(lut1), title('LUT for 4-components')
subplot(2, 3, 2), plot(lut2), title('LUT for 8-components')
subplot(2, 3, 3), imshow(img), title('Input Image')
subplot(2, 3, 4), imshow(imw_4), title("4-neighbors")
subplot(2, 3, 5), imshow(imw_8), title("8-neighbors")
subplot(2, 3, 6), imshow(imw_df), title("Difference")
%% 2. Distance Transform
prob2 = load('prob2.mat');
input_image = prob2.prob2;

mask1 = [Inf 1 Inf; 1 0 Inf; Inf Inf Inf];
output_dist = disttrans(input_image, mask1);
output_dist
%% 3. Fourier descriptor
prob3 = load('prob3.mat');
im1 = prob3.prob3;

[x, y] = size(im1);
lc = length(find(im1==1));

pos = zeros(lc, 2);
[i, j] = find(im1 == 1);
pos(:, 1) = i;
pos(:, 2) = j;

comp = complex(pos(:, 1), pos(:, 2));

f = fft(comp);
f1 = zeros(size(f));
f1(1:2) = f(1:2);

figure, plot(comp, 'o'), title('Input Shape')
figure, plot(ifft(f1), 'o'), title('Output shape')
%% 4. Color Image Processing
pet = imread('pet_noise.png'); % salt & pepper noise
pet_r = medfilt2(pet(:, :, 1), [5, 5]); % apply median filter
pet_g = medfilt2(pet(:, :, 2), [5, 5]);% mask size = [5, 5]
pet_b = medfilt2(pet(:, :, 3), [5, 5]);

petm = cat(3, pet_r, pet_g, pet_b);
figure, subplot(1, 2, 1), imshow(pet), title("Input Image")
subplot(1, 2, 2), imshow(petm), title("Noise removed")
