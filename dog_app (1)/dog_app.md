
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.


    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.


---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 98% of the first 100 images in `human_files` have a detected human face.

11% of the first 100 images in `dog_files` have a detected human face.


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
human_detected_as_human= 0
for human_face in human_files_short:
    if face_detector(human_face) :
        human_detected_as_human += 1
print(human_detected_as_human*1.0/100)

dog_detected_as_human = 0
for dog_face in dog_files_short:
    if face_detector(dog_face) :
        dog_detected_as_human += 1
print(dog_detected_as_human*1.0/100)
```

    0.98
    0.11


__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__ I think it is an reasonable expectatation to pose on the user. We need to create a nearly perfect model and as question clearly states, we risk having unneccessarily frustated users if we don't have clear view of face. So, we can expect users to provide clear view of their faces.

For improvement in face detection we can use some different classifier. One alternative I got from this answer (https://stackoverflow.com/questions/46043103/how-to-improve-face-detection-performance-with-opencv-on-android) is profile face classifier which can increase the accuracy.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 1% of the images in `human_files_short` have a detected dog.

100% of the images in `dog_files_short` have a detected dog.


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
human_detected_as_dogs = 0
for face in human_files_short:
    if dog_detector(face) :
        human_detected_as_dogs += 1
print(human_detected_as_dogs*1.0/100)

dogs_detected_as_dogs = 0
for face in dog_files_short:
    if dog_detector(face) :
        dogs_detected_as_dogs += 1
print(dogs_detected_as_dogs*1.0/100)
```

    0.01
    1.0


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [00:55<00:00, 120.64it/s]
    100%|██████████| 835/835 [00:06<00:00, 138.52it/s]
    100%|██████████| 836/836 [00:06<00:00, 139.20it/s]


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ In CNN, our aim is to decrease the width and increase the depth as we go deep in the network. Conv2D layers increase the depth while MaxPooling2D layers decrease the width. 

 Since, our first layer is convolution layer, we need to provide input shape, but only in first case.
 
`relu` Activation function is used because it solves vanishing gradient problem. 

At the end, a dense layer is added which takes flattened layer by CNN as the input and classifies accordingly. 
I have increased the number of filters in powers of two as described in the lessons.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.
model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='valid', activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, strides=1, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(133))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_7 (Conv2D)            (None, 223, 223, 16)      208       
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 110, 110, 32)      2080      
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 54, 54, 64)        8256      
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 27, 27, 64)        0         
    _________________________________________________________________
    global_average_pooling2d_3 ( (None, 64)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               8645      
    =================================================================
    Total params: 19,189.0
    Trainable params: 19,189.0
    Non-trainable params: 0.0
    _________________________________________________________________


### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 5

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6660/6680 [============================>.] - ETA: 0s - loss: 8.7433 - acc: 0.0075Epoch 00000: val_loss improved from inf to 9.28717, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 140s - loss: 8.7447 - acc: 0.0075 - val_loss: 9.2872 - val_acc: 0.0096
    Epoch 2/5
    6660/6680 [============================>.] - ETA: 0s - loss: 8.3227 - acc: 0.0095Epoch 00001: val_loss improved from 9.28717 to 8.28103, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 141s - loss: 8.3098 - acc: 0.0094 - val_loss: 8.2810 - val_acc: 0.0096
    Epoch 3/5
    6660/6680 [============================>.] - ETA: 0s - loss: 8.2236 - acc: 0.0093Epoch 00002: val_loss did not improve
    6680/6680 [==============================] - 140s - loss: 8.2328 - acc: 0.0093 - val_loss: 8.2810 - val_acc: 0.0096
    Epoch 4/5
    6660/6680 [============================>.] - ETA: 0s - loss: 8.2309 - acc: 0.0093Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 140s - loss: 8.2328 - acc: 0.0093 - val_loss: 8.2810 - val_acc: 0.0096
    Epoch 5/5
    6660/6680 [============================>.] - ETA: 0s - loss: 8.2285 - acc: 0.0093Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 141s - loss: 8.2328 - acc: 0.0093 - val_loss: 8.2810 - val_acc: 0.0096





    <keras.callbacks.History at 0x7f91e01f4ef0>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 0.9569%


---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_4 ( (None, 512)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229.0
    Trainable params: 68,229.0
    Non-trainable params: 0.0
    _________________________________________________________________


### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=30, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/30
    6620/6680 [============================>.] - ETA: 0s - loss: 7.8440 - acc: 0.5048Epoch 00000: val_loss improved from inf to 8.51552, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.8463 - acc: 0.5046 - val_loss: 8.5155 - val_acc: 0.4156
    Epoch 2/30
    6540/6680 [============================>.] - ETA: 0s - loss: 7.7739 - acc: 0.5106Epoch 00001: val_loss improved from 8.51552 to 8.43289, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.7798 - acc: 0.5103 - val_loss: 8.4329 - val_acc: 0.4144
    Epoch 3/30
    6560/6680 [============================>.] - ETA: 0s - loss: 7.7469 - acc: 0.5145Epoch 00002: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 7.7626 - acc: 0.5135 - val_loss: 8.4708 - val_acc: 0.4156
    Epoch 4/30
    6580/6680 [============================>.] - ETA: 0s - loss: 7.7624 - acc: 0.5144Epoch 00003: val_loss improved from 8.43289 to 8.43270, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.7537 - acc: 0.5150 - val_loss: 8.4327 - val_acc: 0.4120
    Epoch 5/30
    6620/6680 [============================>.] - ETA: 0s - loss: 7.7020 - acc: 0.5148Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 7.7005 - acc: 0.5148 - val_loss: 8.4429 - val_acc: 0.4228
    Epoch 6/30
    6620/6680 [============================>.] - ETA: 0s - loss: 7.5788 - acc: 0.5211Epoch 00005: val_loss improved from 8.43270 to 8.27615, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.5976 - acc: 0.5201 - val_loss: 8.2761 - val_acc: 0.4335
    Epoch 7/30
    6560/6680 [============================>.] - ETA: 0s - loss: 7.5532 - acc: 0.5238Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 7.5275 - acc: 0.5253 - val_loss: 8.3005 - val_acc: 0.4311
    Epoch 8/30
    6660/6680 [============================>.] - ETA: 0s - loss: 7.3897 - acc: 0.5276Epoch 00007: val_loss improved from 8.27615 to 8.22254, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.3941 - acc: 0.5274 - val_loss: 8.2225 - val_acc: 0.4275
    Epoch 9/30
    6500/6680 [============================>.] - ETA: 0s - loss: 7.2034 - acc: 0.5375Epoch 00008: val_loss improved from 8.22254 to 8.16111, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.2147 - acc: 0.5371 - val_loss: 8.1611 - val_acc: 0.4359
    Epoch 10/30
    6600/6680 [============================>.] - ETA: 0s - loss: 7.0902 - acc: 0.5477Epoch 00009: val_loss improved from 8.16111 to 8.03250, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.0941 - acc: 0.5476 - val_loss: 8.0325 - val_acc: 0.4383
    Epoch 11/30
    6500/6680 [============================>.] - ETA: 0s - loss: 7.0179 - acc: 0.5574Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 7.0415 - acc: 0.5557 - val_loss: 8.0426 - val_acc: 0.4371
    Epoch 12/30
    6640/6680 [============================>.] - ETA: 0s - loss: 7.0189 - acc: 0.5601Epoch 00011: val_loss improved from 8.03250 to 8.00145, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.0256 - acc: 0.5596 - val_loss: 8.0015 - val_acc: 0.4371
    Epoch 13/30
    6500/6680 [============================>.] - ETA: 0s - loss: 7.0216 - acc: 0.5609Epoch 00012: val_loss improved from 8.00145 to 7.91663, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.0182 - acc: 0.5611 - val_loss: 7.9166 - val_acc: 0.4491
    Epoch 14/30
    6640/6680 [============================>.] - ETA: 0s - loss: 6.9343 - acc: 0.5622Epoch 00013: val_loss improved from 7.91663 to 7.76767, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.9274 - acc: 0.5626 - val_loss: 7.7677 - val_acc: 0.4491
    Epoch 15/30
    6540/6680 [============================>.] - ETA: 0s - loss: 6.7624 - acc: 0.5735Epoch 00014: val_loss improved from 7.76767 to 7.65872, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.7679 - acc: 0.5734 - val_loss: 7.6587 - val_acc: 0.4635
    Epoch 16/30
    6620/6680 [============================>.] - ETA: 0s - loss: 6.7445 - acc: 0.5770Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 6.7395 - acc: 0.5774 - val_loss: 7.7066 - val_acc: 0.4515
    Epoch 17/30
    6580/6680 [============================>.] - ETA: 0s - loss: 6.7298 - acc: 0.5787Epoch 00016: val_loss improved from 7.65872 to 7.63767, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.7256 - acc: 0.5790 - val_loss: 7.6377 - val_acc: 0.4671
    Epoch 18/30
    6620/6680 [============================>.] - ETA: 0s - loss: 6.6652 - acc: 0.5807Epoch 00017: val_loss improved from 7.63767 to 7.51748, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.6609 - acc: 0.5810 - val_loss: 7.5175 - val_acc: 0.4731
    Epoch 19/30
    6620/6680 [============================>.] - ETA: 0s - loss: 6.4992 - acc: 0.5884Epoch 00018: val_loss improved from 7.51748 to 7.42600, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.4926 - acc: 0.5888 - val_loss: 7.4260 - val_acc: 0.4635
    Epoch 20/30
    6520/6680 [============================>.] - ETA: 0s - loss: 6.4605 - acc: 0.5954Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 6.4272 - acc: 0.5975 - val_loss: 7.4599 - val_acc: 0.4719
    Epoch 21/30
    6540/6680 [============================>.] - ETA: 0s - loss: 6.4252 - acc: 0.5994Epoch 00020: val_loss improved from 7.42600 to 7.41869, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 6.4186 - acc: 0.5997 - val_loss: 7.4187 - val_acc: 0.4754
    Epoch 22/30
    6500/6680 [============================>.] - ETA: 0s - loss: 6.4198 - acc: 0.5982Epoch 00021: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 6.4037 - acc: 0.5993 - val_loss: 7.4548 - val_acc: 0.4695
    Epoch 23/30
    6600/6680 [============================>.] - ETA: 0s - loss: 6.3620 - acc: 0.6023Epoch 00022: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 6.3678 - acc: 0.6019 - val_loss: 7.4214 - val_acc: 0.4754
    Epoch 24/30
    6640/6680 [============================>.] - ETA: 0s - loss: 6.3599 - acc: 0.6021Epoch 00023: val_loss improved from 7.41869 to 7.40771, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.3604 - acc: 0.6021 - val_loss: 7.4077 - val_acc: 0.4862
    Epoch 25/30
    6580/6680 [============================>.] - ETA: 0s - loss: 6.3494 - acc: 0.6040Epoch 00024: val_loss improved from 7.40771 to 7.36820, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.3508 - acc: 0.6039 - val_loss: 7.3682 - val_acc: 0.4862
    Epoch 26/30
    6500/6680 [============================>.] - ETA: 0s - loss: 6.3613 - acc: 0.6032Epoch 00025: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 6.3469 - acc: 0.6042 - val_loss: 7.4553 - val_acc: 0.4838
    Epoch 27/30
    6640/6680 [============================>.] - ETA: 0s - loss: 6.3458 - acc: 0.6054Epoch 00026: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 6.3488 - acc: 0.6052 - val_loss: 7.3764 - val_acc: 0.4731
    Epoch 28/30
    6600/6680 [============================>.] - ETA: 0s - loss: 6.3521 - acc: 0.6047Epoch 00027: val_loss improved from 7.36820 to 7.35539, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 6.3460 - acc: 0.6051 - val_loss: 7.3554 - val_acc: 0.4898
    Epoch 29/30
    6560/6680 [============================>.] - ETA: 0s - loss: 6.3365 - acc: 0.6062Epoch 00028: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 6.3457 - acc: 0.6057 - val_loss: 7.3957 - val_acc: 0.4922
    Epoch 30/30
    6560/6680 [============================>.] - ETA: 0s - loss: 6.3423 - acc: 0.6059Epoch 00029: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 6.3447 - acc: 0.6057 - val_loss: 7.3866 - val_acc: 0.4874





    <keras.callbacks.History at 0x7f90ec324080>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 49.2823%


### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features_new = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_Inception = bottleneck_features_new['train']
valid_Inception = bottleneck_features_new['valid']
test_Inception = bottleneck_features_new['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ I'm using Inception model.
 
Before final dense layer, I have added a dropout layer to enhance model training speed and reduce overfitting. Final layer is a fully connected dense layer, activated by softmax function, which ensures that a output of network is a value between 0 and 1, which can be treated as probability of that output.

I also tried VGG19 model which gave nearly 60% accuracy while Inception model gave around 80% test accuracy.



```python
### TODO: Define your architecture.
Inception_model = Sequential()
Inception_model.add(GlobalAveragePooling2D(input_shape=train_Inception.shape[1:]))
Inception_model.add(Dropout(0.2))
Inception_model.add(Dense(133, activation='softmax'))
```

### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
Inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Inception.hdf5', 
                               verbose=1, save_best_only=True)

Inception_model.fit(train_Inception, train_targets, 
          validation_data=(valid_Inception, valid_targets),
          epochs=50, batch_size=30, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/50
    6630/6680 [============================>.] - ETA: 0s - loss: 1.2676 - acc: 0.6899Epoch 00000: val_loss improved from inf to 0.59850, saving model to saved_models/weights.best.Inception.hdf5
    6680/6680 [==============================] - 3s - loss: 1.2624 - acc: 0.6904 - val_loss: 0.5985 - val_acc: 0.7964
    Epoch 2/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.4913 - acc: 0.8485Epoch 00001: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.4929 - acc: 0.8479 - val_loss: 0.6277 - val_acc: 0.8395
    Epoch 3/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.3779 - acc: 0.8841Epoch 00002: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.3777 - acc: 0.8841 - val_loss: 0.6216 - val_acc: 0.8455
    Epoch 4/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.3132 - acc: 0.8967Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.3137 - acc: 0.8966 - val_loss: 0.6231 - val_acc: 0.8383
    Epoch 5/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.2613 - acc: 0.9154Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.2619 - acc: 0.9150 - val_loss: 0.6506 - val_acc: 0.8515
    Epoch 6/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.2337 - acc: 0.9260Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.2348 - acc: 0.9254 - val_loss: 0.6460 - val_acc: 0.8575
    Epoch 7/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.1965 - acc: 0.9377Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1962 - acc: 0.9377 - val_loss: 0.6343 - val_acc: 0.8551
    Epoch 8/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.1784 - acc: 0.9442Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1790 - acc: 0.9439 - val_loss: 0.6919 - val_acc: 0.8587
    Epoch 9/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.1514 - acc: 0.9534Epoch 00008: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1512 - acc: 0.9534 - val_loss: 0.6893 - val_acc: 0.8467
    Epoch 10/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.1441 - acc: 0.9528Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1434 - acc: 0.9530 - val_loss: 0.6901 - val_acc: 0.8575
    Epoch 11/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.1256 - acc: 0.9602Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1264 - acc: 0.9597 - val_loss: 0.7183 - val_acc: 0.8479
    Epoch 12/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.1189 - acc: 0.9622Epoch 00011: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1187 - acc: 0.9623 - val_loss: 0.7299 - val_acc: 0.8575
    Epoch 13/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.1047 - acc: 0.9677Epoch 00012: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.1035 - acc: 0.9681 - val_loss: 0.7134 - val_acc: 0.8587
    Epoch 14/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0914 - acc: 0.9715Epoch 00013: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0912 - acc: 0.9714 - val_loss: 0.7735 - val_acc: 0.8563
    Epoch 15/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0856 - acc: 0.9723Epoch 00014: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0863 - acc: 0.9723 - val_loss: 0.7390 - val_acc: 0.8527
    Epoch 16/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.0803 - acc: 0.9718Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0807 - acc: 0.9716 - val_loss: 0.7808 - val_acc: 0.8527
    Epoch 17/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0709 - acc: 0.9772Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0706 - acc: 0.9772 - val_loss: 0.7951 - val_acc: 0.8515
    Epoch 18/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0694 - acc: 0.9765Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0700 - acc: 0.9763 - val_loss: 0.8085 - val_acc: 0.8563
    Epoch 19/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0602 - acc: 0.9789Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0598 - acc: 0.9790 - val_loss: 0.7788 - val_acc: 0.8491
    Epoch 20/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0582 - acc: 0.9824Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0581 - acc: 0.9823 - val_loss: 0.8073 - val_acc: 0.8599
    Epoch 21/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.0551 - acc: 0.9833Epoch 00020: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0556 - acc: 0.9831 - val_loss: 0.8221 - val_acc: 0.8659
    Epoch 22/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.0506 - acc: 0.9854Epoch 00021: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0503 - acc: 0.9855 - val_loss: 0.8300 - val_acc: 0.8599
    Epoch 23/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0457 - acc: 0.9847Epoch 00022: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0455 - acc: 0.9847 - val_loss: 0.8688 - val_acc: 0.8515
    Epoch 24/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.0453 - acc: 0.9848Epoch 00023: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0467 - acc: 0.9843 - val_loss: 0.8748 - val_acc: 0.8575
    Epoch 25/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0404 - acc: 0.9888Epoch 00024: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0404 - acc: 0.9886 - val_loss: 0.8648 - val_acc: 0.8551
    Epoch 26/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0398 - acc: 0.9879Epoch 00025: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0396 - acc: 0.9879 - val_loss: 0.8880 - val_acc: 0.8527
    Epoch 27/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0383 - acc: 0.9878Epoch 00026: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0383 - acc: 0.9877 - val_loss: 0.9125 - val_acc: 0.8575
    Epoch 28/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.0376 - acc: 0.9892Epoch 00027: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0376 - acc: 0.9892 - val_loss: 0.9258 - val_acc: 0.8539
    Epoch 29/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0307 - acc: 0.9900Epoch 00028: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0309 - acc: 0.9898 - val_loss: 0.9514 - val_acc: 0.8467
    Epoch 30/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0277 - acc: 0.9915Epoch 00029: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0279 - acc: 0.9913 - val_loss: 0.9770 - val_acc: 0.8563
    Epoch 31/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0302 - acc: 0.9899Epoch 00030: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0300 - acc: 0.9900 - val_loss: 0.9091 - val_acc: 0.8563
    Epoch 32/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0269 - acc: 0.9911Epoch 00031: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0272 - acc: 0.9910 - val_loss: 0.9716 - val_acc: 0.8599
    Epoch 33/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0271 - acc: 0.9909Epoch 00032: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0273 - acc: 0.9909 - val_loss: 0.8972 - val_acc: 0.8587
    Epoch 34/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.0281 - acc: 0.9919Epoch 00033: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0277 - acc: 0.9921 - val_loss: 0.9414 - val_acc: 0.8599
    Epoch 35/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.0264 - acc: 0.9916Epoch 00034: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0264 - acc: 0.9916 - val_loss: 0.9341 - val_acc: 0.8587
    Epoch 36/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0213 - acc: 0.9924Epoch 00035: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0211 - acc: 0.9925 - val_loss: 0.9831 - val_acc: 0.8551
    Epoch 37/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0249 - acc: 0.9931Epoch 00036: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0247 - acc: 0.9931 - val_loss: 0.9533 - val_acc: 0.8527
    Epoch 38/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0203 - acc: 0.9935Epoch 00037: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0201 - acc: 0.9936 - val_loss: 0.9861 - val_acc: 0.8563
    Epoch 39/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.0190 - acc: 0.9940Epoch 00038: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0193 - acc: 0.9939 - val_loss: 1.0299 - val_acc: 0.8599
    Epoch 40/50
    6660/6680 [============================>.] - ETA: 0s - loss: 0.0237 - acc: 0.9926Epoch 00039: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0236 - acc: 0.9927 - val_loss: 0.9911 - val_acc: 0.8551
    Epoch 41/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0202 - acc: 0.9941Epoch 00040: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0201 - acc: 0.9942 - val_loss: 0.9736 - val_acc: 0.8551
    Epoch 42/50
    6570/6680 [============================>.] - ETA: 0s - loss: 0.0162 - acc: 0.9944Epoch 00041: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0159 - acc: 0.9945 - val_loss: 1.0172 - val_acc: 0.8563
    Epoch 43/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0165 - acc: 0.9941Epoch 00042: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0167 - acc: 0.9940 - val_loss: 1.0309 - val_acc: 0.8527
    Epoch 44/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0159 - acc: 0.9955Epoch 00043: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0158 - acc: 0.9955 - val_loss: 1.0345 - val_acc: 0.8575
    Epoch 45/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0176 - acc: 0.9936Epoch 00044: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0174 - acc: 0.9937 - val_loss: 1.0323 - val_acc: 0.8611
    Epoch 46/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0155 - acc: 0.9947Epoch 00045: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0157 - acc: 0.9946 - val_loss: 1.0633 - val_acc: 0.8551
    Epoch 47/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0148 - acc: 0.9958Epoch 00046: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0147 - acc: 0.9958 - val_loss: 1.0281 - val_acc: 0.8587
    Epoch 48/50
    6600/6680 [============================>.] - ETA: 0s - loss: 0.0164 - acc: 0.9953Epoch 00047: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0163 - acc: 0.9954 - val_loss: 1.0950 - val_acc: 0.8515
    Epoch 49/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0129 - acc: 0.9956Epoch 00048: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0128 - acc: 0.9957 - val_loss: 1.0697 - val_acc: 0.8503
    Epoch 50/50
    6630/6680 [============================>.] - ETA: 0s - loss: 0.0157 - acc: 0.9943Epoch 00049: val_loss did not improve
    6680/6680 [==============================] - 3s - loss: 0.0159 - acc: 0.9942 - val_loss: 1.0612 - val_acc: 0.8491





    <keras.callbacks.History at 0x7f8fb80cce10>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
VGG19_model.load_weights('saved_models/weights.best.Inception.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.

# get index of predicted dog breed for each image in test set
Inception_predictions = [np.argmax(Inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Inception]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Inception_predictions)==np.argmax(test_targets, axis=1))/len(Inception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 81.8182%


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *
def Inception_predict_breed(img_path):
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    predicted_vector = Inception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
def predict_breed(image_path):
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    if dog_detector(image_path):
        print("Hey Dog you are .. ")
        return Inception_predict_breed(image_path)
    if face_detector(image_path):
        print("Hey you are human but you look like : ")
        return Inception_predict_breed(image_path)
    raise Exception("You are neither a dog nor a human!!")
    
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ With around 80% test accuracy, I'm satisfied with my algorithm. Given a picture it is correctly classifying its breed. It is working well with dog, human and neither human nor dog images as well. As it was expected that model have to give atleast 60% test accuracy VGG19 had nearly 60% accuracy(classifying 3 out of 5 correctly) while Inception is giving more than 80%(4 out of 5). So, I'm satisfied with my model.

To further improve the model:
* As already suggested, we can use better face detector.
* Fine tune pre-trained networks.
* Also we can use augmentation to improve performance.


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
try:
    print(predict_breed('test_images/dog1.jpg'))
except:
    print("You are neither a dog nor a human")
    
# Correct classification: Beauceron
```


![png](output_65_0.png)


    Hey you are .. 
    Beauceron



```python
try:
    print(predict_breed('test_images/dog2.jpg'))
except:
    print("You are neither a dog nor a human")

# Correct classification: Labrador_retriever
```


![png](output_66_0.png)


    Hey you are .. 
    Labrador_retriever



```python
try:
    print(predict_breed('test_images/dog3.jpg'))
except:
    print("You are neither a dog nor a human")
    
# Correct classification: Golden_retriever
```


![png](output_67_0.png)


    Hey you are .. 
    Golden_retriever



```python
try:
    print(predict_breed('test_images/human1.jpg'))
except:
    print("You are neither a dog nor a human")
    
# Correct classification: human
```


![png](output_68_0.png)


    Hey you are human but you look like : 
    Dachshund



```python
try:
    print(predict_breed('test_images/human2.jpg'))
except:
    print("You are neither a dog nor a human")
    
# Correct classification: human(s)
```


![png](output_69_0.png)


    Hey you are human but you look like : 
    Icelandic_sheepdog



```python
try:
    print(predict_breed('test_images/cat1.jpg'))
except:
    print("You are neither a dog nor a human")
    
# Correct classification: neither human nor dog
```


![png](output_70_0.png)


    You are neither a dog nor a human



```python

```
