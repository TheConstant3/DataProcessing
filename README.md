# preprocessing-for-object-detection

To crop large images in smaller fragments and rewrite annotations for training object detection CNN you need to clone this repo, install requirements and clone repo [DataAugmentationForObjectDetection](https://github.com/Paperspace/DataAugmentationForObjectDetection).

Annotations are write in csv format:
```'path', 'x_min', 'y_min', 'x_max', 'y_max', 'label'```

Additionaly you can use augmentation by rotating, [dafod](https://github.com/Paperspace/DataAugmentationForObjectDetection) methods, noise overlay and adding generated objects (using your custom VAE).

To run preprocess:
```
python preprocess_data.py {size frags} {intersection frags} -r -d -l -v(after add VAEs in dir 'vaes/')
```

To test the programm, large images and annotations are already located in 'Photos/Large/' and 'Annotations/' directories.

Also you can debug annotations on fragments by using debug_annotation.py. You can find example in ```debug_example.ipynb```


