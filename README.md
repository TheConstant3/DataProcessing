# preprocessing-for-object-detection

To crop large image in smaller fragments and save annotations for object detection you need to clone this repo, install requirements and clone repo [DataAugmentationForObjectDetection](https://github.com/Paperspace/DataAugmentationForObjectDetection).

Additionaly you can use augmentation by rotating, [dafod](https://github.com/Paperspace/DataAugmentationForObjectDetection) methods, noise overlay and adding generated objects (using your custom VAE).

To run preprocess :

python preprocess_data.py {size frags} {intersection frags} -r -d -l -v(after add VAEs in dir 'vaes/')

To test the programm large images and annotations are already located in 'Photos/Large/' and 'Annotations/' directories.

Also you can debug annotations by using debug_annotation.py

