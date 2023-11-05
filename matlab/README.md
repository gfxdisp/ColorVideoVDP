# ColorVideoVDP Matlab wrapper

There is no Matlab implementation of ColorVideoVDP, but ColorVideoVDP can be run from Matlab using a wrapper class. The wrapper class allows you to pass Matlab matrices with images/video instead of passing file names. 

To pass an image, you need to pass (height,width,3) or (height,width) matrices.
To pass a video, you need to pass (height,width,3,N) or (height,width,N) matrices, where N is the number of frames. N must not be 3. 

To make it work, you need to have a conda environment with ColorVideoVDP installed. 

Below is an example of running ColorVideoVDP from Matlab. 
```
v = cvvdp( 'cvvdp' ); % 'cvvdp' string is the name of the conda environment
                     
img_ref = imread( '../example_media/wavy_facade.png' );
img_test = imnoise( img_ref, 'gaussian', 0, 0.001 );

[Q, hmap] = v.cmp( img_test, img_ref, 'standard_fhd', 'verbose', true );
```
