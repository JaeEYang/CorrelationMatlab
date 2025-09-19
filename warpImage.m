function warpedImg = warpImage(img, M, refSize)
% Transform image using 3x3 matrix M
% img: input image
% M: 3x3 transform
% refSize: [height width] of output reference (usually TEM image size)
tform = affine2d(M(1:2,1:3)');  % MATLAB expects 2x3, transpose
warpedImg = imwarp(img, tform, 'OutputView', imref2d(refSize));
end