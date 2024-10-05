# SpectaGen
Generate spectacles using Generative AI  
Technologies Used: PyTorch, NextJS


## Dataset Creation
Dataset was created manually by taking spectacle images from [Lenskart.com](https://lenskart.com). All the images after fetching from Lenskart were of the shape - (3, 301, 628). This shape is in the form (Channel, Height, Width) 
Preprocessing of the images were done using [ImageMagick](https://github.com/imagemagick/imagemagick) using the bash script - process_images: 
```
convert "$file" -background white -flatten -resize 512x256\! "specs$counter.jpg"
```
The above preprocessing step changed the shape of all images to (3, 256, 512). The dataset has 1000 images.

