

!pip install tensorflow-gpu
!pip install keras
!pip install segmentation-models
#git clone https://github.com/hoangp/isbi-datasets.git ## run once to get the data set

import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
from scipy.ndimage import rotate
import tensorflow as tf
import segmentation_models as sm


image_names = glob.glob("/content/isbi-datasets/data/images/*.jpg")
image_names.sort()

mask_names = glob.glob("/content/isbi-datasets/data/labels/*.jpg")
mask_names.sort()
images = [cv2.imread(image, 1) for image in image_names] 
image_dataset = np.array(images)
masks = [cv2.imread(mask, 0) for mask in mask_names]
mask_dataset = np.array(masks)


import albumentations as A
images_to_generate=1000


#images_path="membrane/256_patches/images/" #path to original images
#masks_path = "membrane/256_patches/masks/"
images_path='/content/isbi-datasets/data/images/' #path to original images
masks_path = '/content/isbi-datasets/data/labels/'

#img_augmented_path="/content/isbi-datasets/data/new_mg01/" # path to store aumented images
#msk_augmented_path="/content/isbi-datasets/data/new_lb01" # path to store aumented images

directory_img = "new_img1000" # path to store new images
  
directory_lbl = "new_lbl1000"

parent_dir = "/content/isbi-datasets/data/"

  
new_img_path = os.path.join(parent_dir, directory_img)
new_lbl_path = os.path.join(parent_dir, directory_lbl)
if not os.path.exists(new_img_path):
        os.mkdir(new_img_path)

if not os.path.exists(new_lbl_path):
        os.mkdir(new_lbl_path)

images=[] # to store paths of images from folder
masks=[]

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))


aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1)
    ]
)

#random.seed(42)

i=1   # variable to iterate till images_to_generate


while i<=images_to_generate: 
    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]
    #print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = (image_dataset[number])
    original_mask = (mask_dataset[number])
    #original_image = (image)
    #original_mask = (mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']


    new_image_path = "%s/new_image_%s.png" %(new_img_path, i) #generating new images in the new address
    new_label_path = "%s/new_label_%s.png" %(new_lbl_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_label_path, transformed_mask)
    #print(new_image_path, new_label_path)
    
    i =i+1



BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


#Resizing images is optional, CNNs are ok with large images
imsize = 64 #image size
SIZE_X = imsize 
SIZE_Y = imsize

#Capture training image info as a list
train_images = []

for directory_path in glob.glob('/content/isbi-datasets/data/new_img1000'):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      
        train_images.append(img)#resize and change color
train_images = np.array(train_images) #Convert list to array for machine learning processing          

train_masks = [] 
for directory_path in glob.glob('/content/isbi-datasets/data/new_lbl1000'):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)     
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)  
        train_masks.append(mask)
train_masks = np.array(train_masks)

X = train_images #normalization is mandatory here
#Y = train_masks/255
Y = np.where(train_masks/255>=.5, 1.0, 0.0)

#Y = np.expand_dims(Y, axis=3) #not  necessary. 


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
#!pip install segmentation-models

import segmentation_models as sm

#model = sm.Unet(BACKBONE, encoder_weights='imagenet') #was not compatible 
model = sm.Unet('resnet34', encoder_weights=None)

model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)

#print(model.summary())

history=model.fit(x_train, 
          y_train,
          batch_size=8, 
          epochs=10,
          verbose=1,
          validation_data=(x_val, y_val))



#accuracy = model.evaluate(x_val, y_val)
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('/content/isbi-datasets/data/segmentation01.h5')

from tensorflow import keras
model = keras.models.load_model('/content/isbi-datasets/data/segmentation01.h5', compile=False)
#Test on a different image
#READ EXTERNAL IMAGE...
#test_img = cv2.imread('/content/drive/My Drive/Colab Notebooks/data/membrane/train/image/0.png', cv2.IMREAD_COLOR)       
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = fn
  test_img = image.load_img(path, target_size=(imsize, imsize))
  #test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
#test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))

plt.imshow(test_img, cmap='gray')
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img/255)

#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='gray')