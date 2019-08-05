run('vlfeat/toolbox/vl_setup')

imaddr = 'cropped/n02123159/n02123159_284.JPEG';
origIm = imread(imaddr);
        
if  size(origIm,3)==3
    
    I = single(rgb2gray(origIm));

else

    I = single(origIm);

end


[f,d] = vl_dsift(I, 'size', 20, 'step', 10);
origIm = insertMarker(origIm,transpose(f),'x','color','black');
imshow(origIm)