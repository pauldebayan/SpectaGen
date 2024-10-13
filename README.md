# SpectaGen [Live](https://spectagen.web.app)

Generate spectacles using Generative AI  
Technologies Used: PyTorch, HTML, Firebase(for Deployment)


## Dataset Creation
Dataset was created manually by taking spectacle images from [Lenskart.com](https://lenskart.com). All the images after fetching from Lenskart were of the shape - (3, 301, 628). This shape is in the form (Channel, Height, Width).  
Preprocessing of the images were done using [ImageMagick](https://github.com/imagemagick/imagemagick) using the bash script - process_images: 
```
convert "$file" -gravity center -background white -extent 512x512 "specs$counter.jpg"
```
The above preprocessing step changed the shape of all images to (3, 512, 512) i.e. RGB Channel - 3, Height - 512 pixels and Width - 512 pixels. The dataset has 1000 images.

## GANs Algorithm
```
1. Train the Generator:
    (i) Take a mini-batch of random noise vectors and generate a mini-batch of fake images
    (ii) Compute the loss for the generated image by passing  the image to the Discriminator.
    Backpropagate the loss to update the weights for Generator. 
2. Train the Discriminator
    (i) Take a random mini-batch of real images
    (ii) Take a mini-batch of random noise vectors and generate a mini-batch of fake images
    (iii) Compute total loss for both real images and fake images.
    Backpropagate the total error and update weights for the Discriminator
```
The implementation of the above algorithm is done using PyTorch. The dataloder for the custom image dataset is taken from the official PyTorch documentation.

## Training 

The training was done using Google Colab using GPU(Cuda).  
Within 3 epochs the generator was able to generate images of spectacles(with lots of noise). It was trained upto 1000 epochs.

## Deployment

The model is converted to ONNX and deployed to Firebase.
