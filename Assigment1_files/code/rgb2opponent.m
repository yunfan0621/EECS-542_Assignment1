function im_out = rgb2opponent(im)

R  = im(:,:,1);
G  = im(:,:,2);
B  = im(:,:,3);

c1 = (R - G) ./ sqrt(2);
c2 = (R + G - 2*B) ./ sqrt(6);
c3 = (R + G + B) ./ sqrt(3);

im_out = cat(3, c1, c2, c3);

end

