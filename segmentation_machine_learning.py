{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "segmentation machine learning.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyMHmSLxbn5dOu6VjZV2jZVl",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ArashAtibi/Machine-learning/blob/main/segmentation_machine_learning.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7UpzLjkvzFXA",
        "outputId": "5be3fdb7-990f-4ead-dc1e-35b1aa62bca9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting tensorflow-gpu\n",
            "  Downloading tensorflow_gpu-2.7.0-cp37-cp37m-manylinux2010_x86_64.whl (489.6 MB)\n",
            "\u001b[K     |████████████████████████████████| 489.6 MB 21 kB/s \n",
            "\u001b[?25hRequirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.1.0)\n",
            "Requirement already satisfied: tensorflow-estimator<2.8,~=2.7.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (2.7.0)\n",
            "Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.1.2)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.43.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.6.3)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (3.3.0)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.15.0)\n",
            "Requirement already satisfied: wheel<1.0,>=0.32.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (0.37.1)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.13.3)\n",
            "Requirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (1.19.5)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (3.10.0.2)\n",
            "Requirement already satisfied: flatbuffers<3.0,>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (2.0)\n",
            "Requirement already satisfied: absl-py>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (0.12.0)\n",
            "Requirement already satisfied: protobuf>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (3.17.3)\n",
            "Requirement already satisfied: gast<0.5.0,>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (0.4.0)\n",
            "Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (3.1.0)\n",
            "Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (12.0.0)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (0.23.1)\n",
            "Requirement already satisfied: tensorboard~=2.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (2.7.0)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (0.2.0)\n",
            "Requirement already satisfied: keras<2.8,>=2.7.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-gpu) (2.7.0)\n",
            "Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow-gpu) (1.5.2)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (3.3.6)\n",
            "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (0.4.6)\n",
            "Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (57.4.0)\n",
            "Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (0.6.1)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (1.35.0)\n",
            "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (1.8.1)\n",
            "Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (1.0.1)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow-gpu) (2.23.0)\n",
            "Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow-gpu) (4.2.4)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow-gpu) (0.2.8)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow-gpu) (4.8)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow-gpu) (1.3.0)\n",
            "Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard~=2.6->tensorflow-gpu) (4.10.0)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard~=2.6->tensorflow-gpu) (3.7.0)\n",
            "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow-gpu) (0.4.8)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow-gpu) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow-gpu) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow-gpu) (2.10)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow-gpu) (2021.10.8)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow-gpu) (3.1.1)\n",
            "Installing collected packages: tensorflow-gpu\n",
            "Successfully installed tensorflow-gpu-2.7.0\n",
            "Requirement already satisfied: keras in /usr/local/lib/python3.7/dist-packages (2.7.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install tensorflow-gpu\n",
        "!pip install keras"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/hoangp/isbi-datasets.git ## run once to get the data set"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VNETZ_npzb9J",
        "outputId": "26b03dad-96c5-454f-e135-b98d1b22aadc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'isbi-datasets'...\n",
            "remote: Enumerating objects: 105, done.\u001b[K\n",
            "remote: Total 105 (delta 0), reused 0 (delta 0), pack-reused 105\u001b[K\n",
            "Receiving objects: 100% (105/105), 35.19 MiB | 28.97 MiB/s, done.\n",
            "Resolving deltas: 100% (9/9), done.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import glob\n",
        "import cv2\n",
        "import numpy as np\n",
        "from matplotlib import pyplot as plt\n",
        "image_names = glob.glob(\"/content/isbi-datasets/data/images/*.jpg\")\n",
        "image_names.sort()\n",
        "\n",
        "mask_names = glob.glob(\"/content/isbi-datasets/data/labels/*.jpg\")\n",
        "mask_names.sort()\n",
        "images = [cv2.imread(image, 1) for image in image_names] \n",
        "image_dataset = np.array(images)\n",
        "masks = [cv2.imread(mask, 0) for mask in mask_names]\n",
        "mask_dataset = np.array(masks)"
      ],
      "metadata": {
        "id": "6BGyNlLDzf4V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from matplotlib import pyplot as plt\n",
        "from skimage.transform import AffineTransform, warp\n",
        "from skimage import io, img_as_ubyte\n",
        "import random\n",
        "import os\n",
        "from scipy.ndimage import rotate\n",
        "\n",
        "import albumentations as A\n",
        "images_to_generate=1000\n",
        "\n",
        "\n",
        "#images_path=\"membrane/256_patches/images/\" #path to original images\n",
        "#masks_path = \"membrane/256_patches/masks/\"\n",
        "images_path='/content/isbi-datasets/data/images/' #path to original images\n",
        "masks_path = '/content/isbi-datasets/data/labels/'\n",
        "\n",
        "#img_augmented_path=\"/content/isbi-datasets/data/new_mg01/\" # path to store aumented images\n",
        "#msk_augmented_path=\"/content/isbi-datasets/data/new_lb01\" # path to store aumented images\n",
        "\n",
        "directory_img = \"new_img1000\" # path to store new images\n",
        "  \n",
        "directory_lbl = \"new_lbl1000\"\n",
        "\n",
        "parent_dir = \"/content/isbi-datasets/data/\"\n",
        "\n",
        "  \n",
        "new_img_path = os.path.join(parent_dir, directory_img)\n",
        "new_lbl_path = os.path.join(parent_dir, directory_lbl)\n",
        "if not os.path.exists(new_img_path):\n",
        "        os.mkdir(new_img_path)\n",
        "\n",
        "if not os.path.exists(new_lbl_path):\n",
        "        os.mkdir(new_lbl_path)\n",
        "\n",
        "images=[] # to store paths of images from folder\n",
        "masks=[]\n",
        "\n",
        "for im in os.listdir(images_path):  # read image name from folder and append its path into \"images\" array     \n",
        "    images.append(os.path.join(images_path,im))\n",
        "\n",
        "for msk in os.listdir(masks_path):  # read image name from folder and append its path into \"images\" array     \n",
        "    masks.append(os.path.join(masks_path,msk))\n",
        "\n",
        "\n",
        "aug = A.Compose([\n",
        "    A.VerticalFlip(p=0.5),              \n",
        "    A.RandomRotate90(p=0.5),\n",
        "    A.HorizontalFlip(p=1),\n",
        "    A.Transpose(p=1),\n",
        "    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),\n",
        "    A.GridDistortion(p=1)\n",
        "    ]\n",
        ")\n",
        "\n",
        "#random.seed(42)\n",
        "\n",
        "i=1   # variable to iterate till images_to_generate\n",
        "\n",
        "\n",
        "while i<=images_to_generate: \n",
        "    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask\n",
        "    image = images[number]\n",
        "    mask = masks[number]\n",
        "    #print(image, mask)\n",
        "    #image=random.choice(images) #Randomly select an image name\n",
        "    original_image = (image_dataset[number])\n",
        "    original_mask = (mask_dataset[number])\n",
        "    #original_image = (image)\n",
        "    #original_mask = (mask)\n",
        "    \n",
        "    augmented = aug(image=original_image, mask=original_mask)\n",
        "    transformed_image = augmented['image']\n",
        "    transformed_mask = augmented['mask']\n",
        "\n",
        "\n",
        "    new_image_path = \"%s/new_image_%s.png\" %(new_img_path, i) #generating new images in the new address\n",
        "    new_label_path = \"%s/new_label_%s.png\" %(new_lbl_path, i)\n",
        "    io.imsave(new_image_path, transformed_image)\n",
        "    io.imsave(new_label_path, transformed_mask)\n",
        "    #print(new_image_path, new_label_path)\n",
        "    \n",
        "    i =i+1"
      ],
      "metadata": {
        "id": "OPT_Qw4fzqWU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "!pip install segmentation-models\n",
        "import segmentation_models as sm\n",
        "import glob\n",
        "import cv2\n",
        "import os\n",
        "import numpy as np\n",
        "from matplotlib import pyplot as plt\n",
        "\n",
        "BACKBONE = 'resnet34'\n",
        "preprocess_input = sm.get_preprocessing(BACKBONE)\n",
        "\n",
        "\n",
        "#Resizing images is optional, CNNs are ok with large images\n",
        "imsize = 64 #image size\n",
        "SIZE_X = imsize \n",
        "SIZE_Y = imsize\n",
        "\n",
        "#Capture training image info as a list\n",
        "train_images = []\n",
        "\n",
        "for directory_path in glob.glob('/content/isbi-datasets/data/new_img1000'):\n",
        "    for img_path in glob.glob(os.path.join(directory_path, \"*.png\")):\n",
        "        img = cv2.imread(img_path, cv2.IMREAD_COLOR) \n",
        "        img = cv2.resize(img, (SIZE_Y, SIZE_X))\n",
        "        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      \n",
        "        train_images.append(img)#resize and change color\n",
        "train_images = np.array(train_images) #Convert list to array for machine learning processing          \n",
        "\n",
        "train_masks = [] \n",
        "for directory_path in glob.glob('/content/isbi-datasets/data/new_lbl1000'):\n",
        "    for mask_path in glob.glob(os.path.join(directory_path, \"*.png\")):\n",
        "        mask = cv2.imread(mask_path, 0)     \n",
        "        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))\n",
        "        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)  \n",
        "        train_masks.append(mask)\n",
        "train_masks = np.array(train_masks)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vcKcChDQz1_t",
        "outputId": "48df360c-e7ff-472e-bdaa-391fb795fa52"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: segmentation-models in /usr/local/lib/python3.7/dist-packages (1.0.1)\n",
            "Requirement already satisfied: image-classifiers==1.0.0 in /usr/local/lib/python3.7/dist-packages (from segmentation-models) (1.0.0)\n",
            "Requirement already satisfied: keras-applications<=1.0.8,>=1.0.7 in /usr/local/lib/python3.7/dist-packages (from segmentation-models) (1.0.8)\n",
            "Requirement already satisfied: efficientnet==1.0.0 in /usr/local/lib/python3.7/dist-packages (from segmentation-models) (1.0.0)\n",
            "Requirement already satisfied: scikit-image in /usr/local/lib/python3.7/dist-packages (from efficientnet==1.0.0->segmentation-models) (0.18.3)\n",
            "Requirement already satisfied: h5py in /usr/local/lib/python3.7/dist-packages (from keras-applications<=1.0.8,>=1.0.7->segmentation-models) (3.1.0)\n",
            "Requirement already satisfied: numpy>=1.9.1 in /usr/local/lib/python3.7/dist-packages (from keras-applications<=1.0.8,>=1.0.7->segmentation-models) (1.19.5)\n",
            "Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py->keras-applications<=1.0.8,>=1.0.7->segmentation-models) (1.5.2)\n",
            "Requirement already satisfied: networkx>=2.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (2.6.3)\n",
            "Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (1.2.0)\n",
            "Requirement already satisfied: imageio>=2.3.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (2.4.1)\n",
            "Requirement already satisfied: scipy>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (1.4.1)\n",
            "Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (2021.11.2)\n",
            "Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (3.2.2)\n",
            "Requirement already satisfied: pillow!=7.1.0,!=7.1.1,>=4.3.0 in /usr/local/lib/python3.7/dist-packages (from scikit-image->efficientnet==1.0.0->segmentation-models) (7.1.2)\n",
            "Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->efficientnet==1.0.0->segmentation-models) (3.0.6)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->efficientnet==1.0.0->segmentation-models) (1.3.2)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->efficientnet==1.0.0->segmentation-models) (0.11.0)\n",
            "Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image->efficientnet==1.0.0->segmentation-models) (2.8.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib!=3.0.0,>=2.0.0->scikit-image->efficientnet==1.0.0->segmentation-models) (1.15.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = train_images #normalization is mandatory here\n",
        "#Y = train_masks/255\n",
        "Y = np.where(train_masks/255>=.5, 1.0, 0.0)\n",
        "\n",
        "#Y = np.expand_dims(Y, axis=3) #not  necessary. \n",
        "\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)\n",
        "\n",
        "# preprocess input\n",
        "x_train = preprocess_input(x_train)\n",
        "x_val = preprocess_input(x_val)"
      ],
      "metadata": {
        "id": "hUkQmOPhz8ky"
      },
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# define model\n",
        "#!pip install segmentation-models\n",
        "\n",
        "import segmentation_models as sm\n",
        "\n",
        "#model = sm.Unet(BACKBONE, encoder_weights='imagenet') #was not compatible \n",
        "model = sm.Unet('resnet34', encoder_weights=None)\n",
        "\n",
        "model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],)\n",
        "\n",
        "#print(model.summary())"
      ],
      "metadata": {
        "id": "0av0Xr--0Aad"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history=model.fit(x_train, \n",
        "          y_train,\n",
        "          batch_size=8, \n",
        "          epochs=10,\n",
        "          verbose=1,\n",
        "          validation_data=(x_val, y_val))\n",
        "\n",
        "\n",
        "\n",
        "#accuracy = model.evaluate(x_val, y_val)\n",
        "#plot the training and validation accuracy and loss at each epoch\n",
        "loss = history.history['loss']\n",
        "val_loss = history.history['val_loss']\n",
        "epochs = range(1, len(loss) + 1)\n",
        "plt.plot(epochs, loss, 'y', label='Training loss')\n",
        "plt.plot(epochs, val_loss, 'r', label='Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "plt.xlabel('Epochs')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 697
        },
        "id": "HGp8Qq6M0EVw",
        "outputId": "3f3ac93b-38e6-47c6-951c-4cf25ca666ce"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "100/100 [==============================] - 206s 2s/step - loss: 0.8430 - iou_score: 0.6974 - val_loss: 0.8502 - val_iou_score: 0.6970\n",
            "Epoch 2/10\n",
            "100/100 [==============================] - 205s 2s/step - loss: 0.8428 - iou_score: 0.6973 - val_loss: 0.8465 - val_iou_score: 0.7022\n",
            "Epoch 3/10\n",
            "100/100 [==============================] - 204s 2s/step - loss: 0.8422 - iou_score: 0.6976 - val_loss: 0.8665 - val_iou_score: 0.7029\n",
            "Epoch 4/10\n",
            "100/100 [==============================] - 206s 2s/step - loss: 0.8404 - iou_score: 0.6978 - val_loss: 0.8641 - val_iou_score: 0.7046\n",
            "Epoch 5/10\n",
            "100/100 [==============================] - 207s 2s/step - loss: 0.8374 - iou_score: 0.6988 - val_loss: 0.8497 - val_iou_score: 0.7059\n",
            "Epoch 6/10\n",
            "100/100 [==============================] - 207s 2s/step - loss: 0.8342 - iou_score: 0.6993 - val_loss: 0.8516 - val_iou_score: 0.7057\n",
            "Epoch 7/10\n",
            "100/100 [==============================] - 203s 2s/step - loss: 0.8330 - iou_score: 0.6996 - val_loss: 1.6830 - val_iou_score: 0.7454\n",
            "Epoch 8/10\n",
            "100/100 [==============================] - 206s 2s/step - loss: 0.8268 - iou_score: 0.7008 - val_loss: 0.8561 - val_iou_score: 0.7033\n",
            "Epoch 9/10\n",
            "100/100 [==============================] - 207s 2s/step - loss: 0.8204 - iou_score: 0.7028 - val_loss: 0.8618 - val_iou_score: 0.6988\n",
            "Epoch 10/10\n",
            "100/100 [==============================] - 204s 2s/step - loss: 0.8194 - iou_score: 0.7026 - val_loss: 1.0717 - val_iou_score: 0.7181\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:1410: CustomMaskWarning: Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.\n",
            "  layer_config = serialize_layer_fn(layer)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('/content/isbi-datasets/data/segmentation01.h5')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 166
        },
        "id": "aW4EzgfVfcEd",
        "outputId": "35e90208-9b76-47a3-db02-3c5f303afd54"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-2-2047d6afcc2c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msave\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/isbi-datasets/data/segmentation09.h5'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'model' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow import keras\n",
        "model = keras.models.load_model('/content/isbi-datasets/data/segmentation01.h5', compile=False)\n",
        "#Test on a different image\n",
        "#READ EXTERNAL IMAGE...\n",
        "#test_img = cv2.imread('/content/drive/My Drive/Colab Notebooks/data/membrane/train/image/0.png', cv2.IMREAD_COLOR)       \n",
        "import numpy as np\n",
        "from google.colab import files\n",
        "from keras.preprocessing import image\n",
        "\n",
        "uploaded = files.upload()\n",
        "\n",
        "for fn in uploaded.keys():\n",
        " \n",
        "  # predicting images\n",
        "  path = fn\n",
        "  test_img = image.load_img(path, target_size=(imsize, imsize))\n",
        "  #test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)\n",
        "#test_img = cv2.resize(test_img, (SIZE_Y, SIZE_X))\n",
        "\n",
        "plt.imshow(test_img, cmap='gray')\n",
        "test_img = np.expand_dims(test_img, axis=0)\n",
        "\n",
        "prediction = model.predict(test_img/255)"
      ],
      "metadata": {
        "id": "6xrcaOLC0LrZ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 381
        },
        "outputId": "d22d46a8-b986-40bd-eabf-5def728a52cf"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "error",
          "ename": "OSError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mOSError\u001b[0m                                   Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-4f2eb6bf9f78>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mkeras\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mkeras\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodels\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/isbi-datasets/data/segmentation01.h5'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcompile\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;31m#Test on a different image\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;31m#READ EXTERNAL IMAGE...\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m#test_img = cv2.imread('/content/drive/My Drive/Colab Notebooks/data/membrane/train/image/0.png', cv2.IMREAD_COLOR)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/keras/utils/traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m     65\u001b[0m     \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# pylint: disable=broad-except\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     66\u001b[0m       \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 67\u001b[0;31m       \u001b[0;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwith_traceback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfiltered_tb\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     68\u001b[0m     \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     69\u001b[0m       \u001b[0;32mdel\u001b[0m \u001b[0mfiltered_tb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.7/dist-packages/keras/saving/save.py\u001b[0m in \u001b[0;36mload_model\u001b[0;34m(filepath, custom_objects, compile, options)\u001b[0m\n\u001b[1;32m    207\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    208\u001b[0m           \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mio\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgfile\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexists\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 209\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mIOError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf'No file or directory found at {filepath}'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    210\u001b[0m           \u001b[0;32mif\u001b[0m \u001b[0msaving_utils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mis_hdf5_filepath\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mh5py\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    211\u001b[0m             raise ImportError(\n",
            "\u001b[0;31mOSError\u001b[0m: No file or directory found at /content/isbi-datasets/data/segmentation01.h5"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#View and Save segmented image\n",
        "prediction_image = prediction.reshape(mask.shape)\n",
        "plt.imshow(prediction_image, cmap='gray')\n"
      ],
      "metadata": {
        "id": "-F6t5W8E0Lo5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 285
        },
        "outputId": "64c72464-54d6-4e38-b093-31f29d1a389a"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f807dcb0790>"
            ]
          },
          "metadata": {},
          "execution_count": 34
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAQEAAAD7CAYAAABqkiE2AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO19a6xt11XeN87rOglp7BDLutihNsKiSlHbhKuQKBWNMJQQHqYSipKi1kAqqxW0PFoRu/yIKhWJtAgIUgu9kECoQh6EtIkiWpq6RKhScXGaNC8nxMSE2HLiawmSKiY+5+wz+2Pvsc+3vzPmXHPtffa5+/qMT9raa635XHPNMeZ4rbmslIJEInF+sXW1O5BIJK4ukgkkEuccyQQSiXOOZAKJxDlHMoFE4pwjmUAicc6xNiZgZq8ws0+Z2UNmds+62kkkEqvB1hEnYGbbAP4YwLcDeATAHwF4TSnlE6feWCKRWAk7a6r3xQAeKqV8BgDM7O0A7gQQMoHrr7++XLx4Edvb2yiloJQCM0OLQXma/x8dHc3P/ZqZVctF51yuJ1+rX0PorW+Z+n38TrNOBdffasvToj4NlWs9x1odtbzL1tG6zzFj0DPGtTp07MY+24ODA0wmEzz88MNPlFJu1DzrYgI3A/gcnT8C4Js5g5ndDeBuALjpppvwq7/6q3j2s5+N/f19TCYT7OxMuzaZTOZljo6OUEqZ/08mE0wmExwdHeGpp57C0dERDg8PcXh4iMlkMmcqBwcH8zoODg5wdHQEM8NkMsHh4eF8UJ966qn5gHvdwHQg9/f353UcHh4uMCrvj/eXHxozK74X76OnORPjezYzHB0dLaTpMdfPE00nytbW1kJZZUKc5tje3p6nm9nCZPbnY2bY3t7G1tbWPJ+X83xeLkrz+r2OqO3t7e15HVtbWyfy+b1ubW3Nj81s4Vzzcf93d3fDfvg9ez2e5te8TjOb36df4z7ysdfr48jjwfV5Hf5cOJ+nR+d87HPiiSeewF/+5V/ie77nez6LAOtiAoMopVwGcBkAbr/99vLkk0/i6OhoTtQXLlwAgDnxORE5oTsBHx4eLjCHo6MjfOUrX5kT9GQywZe//OV5u08++eScyA4ODrC/vz8ftC996UvzQd/f358zj1IKvvKVr8wHdX9//wQj8XzOZHZ2dub3433zfF4/MwFNc6LxOhxen9+bMgE/94kXETDn53q9HBO698Env4/P3t7evM69vb05w93a2pqnAcB1112Hra0tTCaTObF4O3t7e/O+7O7uYmdnZ95fZhC7u7sLxOz5tB/MLDzNceHChXn69vY2dnd35+fXXXfdQh1ev9+r3/ve3h52dnaws7Nzggn4GOzt7S0wCmUMTKRbW1vY3d09wWT4n5+Xgwk+usYMc2dnBzfccAOe85znoIZ1MYFHATyfzm+ZXQuxtbWFZz7zmbhw4cKc0H1SAccD4GlOYLoC+yqxv78/X61dSnAoAe/v788H+8knn5zX5wyGy3nawcHBgiTgE9zT/J68v94PXuGdkbXq8HtnZhFJRkzQ3pbfE+dn1KQPXcF8nHxyeZssqTlh+Jg48Tkz9zp4ZQYwX4H9mW5vb8/7tLu7O78vlSa8HICFldufvz9fZhAXLlxYyOd1eH3M7CKp6ejoaIFgmUhZEvA2WZpgZsHgVb/283w1yUIZAjMB/ldphLEuJvBHAG43s9swJf5XA/j7tcxbW1u4cOEC9vb25oQDLIrSfFP805ve3d2di/Kex4nKic+55NHREQ4ODuYPM1IHfGWM1AFP88kDLIryygT8vjxNVQWfgKxuRMzOwUzA6+W8XnekBuh1Z3jRZDs8PFy4P793T3MCdhUsImC+P78nz+dMi6UQtg/xiqgrpNfv9+/1s0TC0oqnRWqJPgMes8PDw3kfmVn4/GMpo0a0ylyUKSp8nJUp+DWe/9pWSzVUrIUJlFIOzexHAfwegG0Aby6lfLyW31drJnKfKH6d9aQLFy7MCccfJhMYcCzW8kD6ADHR8kqi0gcTujIgroNFc+4Hr/5RmvfTrymzYNuHjO88XdUFv87MUe9Jj1Wi4HsHFhmbqjtMxC4xMIPzvjizZULn/rL058+Q+8WTXpkiqwqsejBzZMavkgD3w+9Hy/LYe39rK/v29vYCk+E0h9orPL8fO5PxY5YweIyU2DlN02tYm02glPK7AH53RP4TFn7mmjX9x/+j1UDFIeXeyml59VEmwKKqX+dJy5MVwMKE4wnlaTrhdbXjMfGViseGJ2NE0DoROE803iotREzAweqGH0eiqx+zvsvjwfepRj19jjx2rWev7Q+tgt6+PkN9ZgydfwpeILjPUR3ePkugyryBYyklarfWjx4GAFxFw6DCVzQmUABzUXNra2uui9dWSOaUfl5KmetyPllZnAMWuTKLoCxpMLOI1AFlAlyX9lFXMyVcva6ICJoZQ5TGBB9JXFEZXZGBRZsK3xNwzCxYV1aJIHo+bHj0sXam688gYqCllAV1wFdRv1c3LnOfnJjYuBhJefoseSHxNpWJ+X2yuM79iJhSxGx8rHieO7PVtrUf0fwYwsYwgVLKXL/0B806HOtTvMpyef/niaUrJXDyoXM5Jgye7JGIyuK9TxpVR7jN6Jj7rlDbiBJVNIbRit6SBGppShA8pqzicH+YKUaEFd0fExwzD3/2LGn4hGfDMNfrBkpmOHxPzsCVSHk8uO/Aom1C3b+qPqqa58ds6OX70AWD5xiDx9zPvS5ezBg8H4cYwsYxATawACdXEL45JjQefL3GD6rWNq9KOiF4MHXiaLo+5Joa0xJVmXnV+ttK03w1CYHvNapfx8JX5cjg6cY0lSy8LnajavvsOVDVjCUBlhAi5s79jtQEZwI12wQTms6ZSDLgNnWh4pVc62BpIvKCRXON21ZJlJnnMtgYJsD6EIvt/tA4H1tpmXuaTV01btnnieQTyCew1u8TBFgkMuWy0aTgh6YqRk0CUMbB6bUVgdFK43sYInK+F5UamNA5tkIlCCZmbY9XbJ3YvkJGRjbvF4vXLGV5W+w+ZC+FmS14KdwNuL+/PydCB4vy3DaAhZgBtVsws/b6XHLhFVrnIc/x1hzxsvrMVAJle4H2n8vWsBFMgFd1Nt5pnohY1AjEnFWNM6zf+z+v2jzgXM7/a9xZV4BaX6N/XWG4zRah60OupetKxNci5qDGwogZ8ESs5XNw/hqDqK3ovKrzmPkYKRNwFZLFdW+HpQm2K/B4O3xB4nb8mBk36+KR1BHNnWhu6zNQVUbHhRcxlmCH6q5hY5iAB5hcd911c0OKB+ywUc/zMwH6Q2Oi5tXNER2rCOppyiAcaovwPrA64vXyP6fxde2HHzNh6INVom/do4rfnNbDBPjYVx0l3ogJeFmVBJioa3EU3o6v9hw74nDVQ1f0nZ2duUTnrmReLd3lzKK/P3cOIWfC9v5FRmKOR+ExZ+8Pz0ev09viOeFjoRGGDm5TmSozHJ4HOs8ibAQTABZFPOX+TJAAQi7Oq7k/CBa5nZFEuqCu3l5WrdI1jqv2Cq1TJ4Cne1t8znmWYQJ83pIC9NjPmRD1xwQdEb5KGE58WpaZABMp98s9QTzhuV/+nNW1ymK4i/IsMWgwDy8W/MxUReF5oIsNLwx+rOpq7Z+hiwS3qc+IGZk+49ZxhI1gAj6gHk9fc82xmMcEx6svMwiVFiLOqq4pYNEYGa3Oep0ljjEPGRhmAlE9Q0zA7yFKrzEGLseEz2WUCXB+veZgJuBErWlexldrVjuilZbTVeUzs7mtwaNHS1l8/wDAgivRx50JOFqBWdf3n/eL56ta7HVeqaTnbXi7zDzUHsKLjY/p7u7ugmpQe7Y1bAwT8BthP6hfHxJp2EIdiY5eHxtjnJhbTEC5MTMBBpdTS+0Qg4hW+0gkZfQwARUL+bjGIDwtIn6/b2UCKgWwJOD9YCahxMFl1Cbjz0eZRY1ped380pka0tgYyeWOjqYh5EzcZrYgTaiE6IyEV+doHKLnxX1i4yIzo0h65OfqzJAjOrWMMvQIG8MEOFySiZVFcr7ZCMpZHawKsHEnYgIqFnIdwGIwiLfD9XE57WeLoFsSxNC91tIiBuD9rK0UuuJwfpaQVOTnMrrC1iQBJhplPsBiABkTIasBeo/MdCOpRo1pKmHUxpOfO0uqimju9KBFqDWpk8tFkps+yxo2hgns7OwsvKSjLpSIAzMRu6GF3zDzulkdUAuvuh+B48kUETSnabnWcXReG4se9DCB6Lx27OeROKmrm0oSSvzqAeBytTRtm9UBJla+puX8mr/xyfYhthPoKu5MzQPVdK8E76NHngJYkBhYiuE09+erm5HBdXibLLn64hXNC2YKXHcv8Ts2hgmwe6fnp6t2ZDvguoHFl4VYEnDwClkj9FWYQC9Y6qihVbeKgzX1osYsagzA62YxnvvJzJX7wuk83vzmohOkqk+q4jERqlri7aq+H91fdG86Zr3Slq68tbwt1KTEWv9qUl5rAahhY5iAv0bMXBCIRXyebK2f51X1wutl5sH1K6FHRKV94InLiNyH2pZec9RETi0XgQmv1mZ0zgTA96UTTIkvyqdjrN4BXmWZqTuDVjUAwFwUV0NYBN5Twsv7JjLKWFyC4GAjHh/db8H/2ZXp+SNbRjRu0Tj6sUo//nz0nl0ycrik02L2io1gAkAstgMnLdcR4UYqg3oMVGLgCcp98DYjqSLqM9COEwAWX6ip1TEWrXIuCus1R2Q4jfJG/3zPulJFbkKWavi61qH6uYq5ns8jQtnQ6OV43wM/r636TExcBy9AZseGwa2trfluRN4/zcfjqYuNo8XY+fnUFjT+53H3fC3psYaNYgLMAGo3Diy+QKSDFZWP1IyI0Gtis+px2h+gTug1Yq2tyr2IJnPUZrTq80SMVqraZKut+toXbyMSldV96Nf8mXL9fF2vRasszwG1W7ANQaG6PUslfh9uK9A0btvb5Trc1sDj4JJI5AGoqRY1iU6ZLj+/XmwEE3BOqrvEcDpwcjLyqquiGNcTMQsV+bU/XhfX30Ow0cNrlatdGxLhovFR6GQaOgdie8TQPflEjIxcKnJrv5WxRP2I5oSnKUOL9mXwvEwwLCX4m4fuIuTV3st5fo4HcDVB1QHvi1/nd1KUsek8VmnLy+n9R2PBqoHW0cJGMAEAJwaDr0dEqfkiD4DWwW1oea6/1YeonN5H7doYDj2WmzOiSRMRfAscdAWcDHTx+vk4sn14296+husq02Dw6tpyzUbg/vCxSiXq+tM5xqu7LiTRj9vXOnRBisaKr7ekyEja47GOxrOGjWECrIsBx5tQ8kPxCaRuQAALtgAeJCb8iDlExz3nY9LGEuBpoSVN1NJ0QjlYvNc0NXZpWqSTa1rNfcj2Ah9Ldhv6Kmt2vHmrp/M+gbz3pBrT3BjoRkOec3t7e/O6fCdhZXzMJFwi2to63gSH1QGdnxoCz3l4LGuLWvTMlGEOYSOYgA88i2G6etYkBa6DV/tItIrE/15G0KsO9NTbW7a24g6pCrV6etKi+xy6b9Xne8cqGqPINuDpOhci9SK6ruUV/jJStDehqwGTyWRueOQYFd4U1sdKGRuL9coolfFxmh7z+dAYRzEJNWwEEwCOdXlgUWRriURaviUJRCKb1sMPMrre6stYEazW1lDfojbG2A80f8QMWkyI8/C1iBEMQdthiY03EOV/ZexMlFHADbelIrS36xIChw17Ga+Xt5L3fnDgkuf3vnCsQuROjTaPjTxike1FmWRLNWgtBI6NYgJRhFe0GrB+5tBrPAEibwPXqcdjVIGhsr3lVkFvHUMGytZEUQKMJhjXoQFEkarAdbKYrXXos2RVQYmBGYP2R42N3n/faITLe32uRpjZ3Dvgadvb2wuvNJstbq3u7UevqKsdhBcrlmQiHT+SiH1M3BgZvU1bw8YwAe1oRFR8LcrPg1RLG0Pgy6b1oEcS6MVY9aBWJiJqzl9jANHqH1n3a+3pKq4Erp4KTlMGw+fMWFzvj/RuJn7/KXOKvE46r/TH16NnwPfDaZHko0w6ImwOxlJbSgsbwQRYfIs4pOZlrq2rQDQ40QOpifytPkbleu5tlfSxbZwGQ4hESJ2sehz9A201QZkGMwx/trqSa4CQPlM2jHEdOgf0nlnnZ8LnHa5dIuD7cUnAmR5vkaevQXsfuf1I9I/mmNfDm/Eqs2AXLUtPQ3N2I5gAsMi1a+oAUOe4PXp/j6QxpNtH109zVV8VQ+23JADN0yMJ1Pqg9gIg3kWXV29lPlqHM3reAShaIbmtqH4Vp1XSUHXA03WTGd62XFUWtforE2L7RzROtbnM83xIMmBcEzYBYDE4IrJs6kOM9P4acUfE3iJoPe7BaUoIY7GMBFCrpzYuPcxB62oxdhb1dcXk6D4Wz3nFA3DC1ccSA7/i7Nc8MEjDfF0S0Lni+Z2wPe/W1tZ8rwtfnbl/7DKMApgiW0ikDnAf2UaijKAlCQ9haSZgZs8H8JsAbgJQAFwupbzRzJ4L4B0AbgXwpwBeVUr584G6Fl7c8Gv878e68kdpEdHr8Vms6Eoc65QSaqv5MvXUykZpNTWB+6QiMF+PXHo1aWWIOen16Dyqw3Vp9w5o3RxhyEzAj1UX97LMyLRdDWtmZqE2Ds0X2Vk0L7czFDPQ50iMcQjgn5dSXgDgJQB+xMxeAOAeAPeVUm4HcN/svBsRAesDVNG/xQC07t4+jO2z9rcliZwFVmnztPtaI8ChdlrlhiS9Wj3RnAEWPRIcgsvnTtQaAKXnXl/rFwVRDZVl1NJa9dSwtCRQSnkMwGOz4/9nZg8CuBnAnQBePsv2FgAfAPC6ofpUGlA9UsWdHmbBaXo8drUfKnfWRN6LVr9aEyMqt4qE5Kskl9dQZPXz62TmOlTP17bUzebtsajuaf56soZKe13qIvR5qBuZOthN6Su82g7Y7sCLm1/j9h0tG4AyAh3LdUkC3KFbAbwQwP0AbpoxCAD4PKbqQlTmbjN7wMweeOKJJ6oPs2UbiK6d9mo/tMpcyxgrLfTk7WHAtTT10bcYe63tqHyLECIpIYLaMfy/Jh1EIn1r5e5R35j59ZbrqXdlw6CZfRWA3wHw46WULwl3KmYW9qKUchnAZQD4pm/6puIPINIR2TVE7Y6acLVrA/c2Kv+1Cl0xlymvq1TrfKjtyK3Iq36rPk7n/16G55IB16WbeboBsZTp1uj6IV2XCtyO4PW6DcQllWgcdCxUguC6vE2tC8AJSWgt6sCs47uYMoC3llLePbv8BTO7WEp5zMwuAni8s675L/KBRp9tioh+LPGfF0LvQW0i1vJwPi27DMNlgmFGoETN+XUO8ByKPqnuRM199V2tvIzuE8DqAM9PJ3b2FPjLRKxeaCQs34eGKPOY6vjqWLNqwSoAl12rJGDTnrwJwIOllJ+npPcCuAvAz87+3zOiziqBRxy9tvIn0a+OaJyiCRUxjqFrUR2R5OBl1GukgURaV7Q66vxhOwHr+s4IovnHQTrMBPSYd7Z2wvc++LF+Eo+Pa9GTOvdrDI/Hbt3qwMsA/AMAHzWzD8+u/UtMif+dZvZaAJ8F8Kqhimo3N5SvVldicxCJ7jWJozZxWR2MRF8GM4fo3QNvh1dTnW+8qrp70CMHneDYRegxBp7fpYFIHXCJljdZ1TmrUbNcvmYs5XcHlIG1QreB1bwD/xNAreY7lq03EnsisY//e+tNrIZeXbx1XZ9Dq85IbObruqpreq+RsCZhar294jXnbRkJOa+qQ72LXFS3tj+EjYkYjCZH7Vrq/VcPQ2LmkM2gJfb3IHoRiNvglY/rZbegn3t/OLrPV1X+orGX46hAVwE0pJjfIvS+6afxVOrV41pEYKTKKJQpTCaThfDmCBvDBBgRpx47WZYpkzgbDEkUmhc4uVpyPcpgosVjLFgdYOagLzHpT/V5Vw+8fyoJREZAFvX1XmtSBbetNpPIaMjYGCbQ0hN5QCK1oFZXMoD1YYzhKSrHz3VIhTiNfvZC2/TIQf3UGkshTOjMBDzikKUO/UVzVQN9tF8RI+F0l170jcsaNoYJAMPW/jQKbh5qzDtK1zw13V4ZRCQWR/quptUMbuqmcyJ2Ed4t/U7o7PLj6+od8DY9r9fL9en7AfpNDAAn4g68z0r8tbcNo5iCFjaKCSiiCZYMYHPRIx3oM1Vxn4/1PwJP+EhM9zY5ZFiJjF1zuoU5i/LRG4AcRsz1MXHyW5BK8NG1aK5H4xAtiuoJueYMgzXxyM9PQ9dLrB9D0kFP+VVUAZU2+Hqki/M5bx7ihrXDw8OF14yZIbCq4OW9Pn4RiVUIs+kn0l2C0H6qmy/qf+t+lBkPSQMbwwRqOC3dMHHtoaUWcHqrnJ+r2KwrKnsb1B3JG5rUrmtbmsbSiObjOjktuq/WfXOeMUbxjWICNWkgJYBrFy0VoTZhWxNYr7OurO8caL5a3f59Q+BYJfA6+NsF2j9XL4Bjl6OHHbvNwOtTG4C3V/tikt6rMhHvh+ZliYaZ0zUlCQzpgMkAEgw1JvK12jyqqQJazsVo/tQYsLhzsb9AxBZ5B1939UD3KeSNRxyRd8D7pe5GfotxWWwME0ixP+EYMxeivEPicg9j8LyO2sYhTNh+jdUL/jHziPR1tfrX7ofz8X+NGVyT3oHIAjpGx0lsHlqEp2mqEtY8B1yH5+W0SHXgrcAc7DnQaD+XBHzDUSZmjyDc3t6ubirCLkL1PgBYiObjV5C9HoZKDbu7uyeYjObXsY5wKpuKnBZaRJ4M4PygZx601MWWUY3z1AyG0Y/166E9+8ai1/Olfe3FNScJDNkEEonTQM0dHenn0T6DXsbdh36NX012OwIbHb09dzlG3gGXVKKXplSNqEkBeq/XlGEQyFX/vKBX96+5CsfME17R1crOor++mKRx+NymqxDsUWCvgW4qErkOo7BefdlJ78GPW69W1+IJImwkE0icH9QYgbq2IhtB5CJjRNGEUagtE2/EBKJ+8T/bErxdZRAsQWhEodfFbkYNbfZ7iNQab7NmFEwmkNh4rMszdNoSJbvkvH7fdMSPoxeI2EXI7xRwvEDUTnQ9GifPH8UR9KgLyQQSG40hd9+y5SLR2lfqIUOhth/l1bZqfa95MPjcoa8pKyIVYEgKAJIJJDYEkZuwljZUD+v3WmdkT+BPjala4Ku5f6xUy5oZ9vf3w41G2VCo17ltdRHquwM8FtGLR7yRaVQmJYHEucQqKkaL6dQMbrryshiu8QXct4hwvWwP86sxi6hfNSQTSCQELDG0CB04+RYhcLxKsxQBYMFm4Ku9fuPA04A4WGio3xpN6MdaFyOZQGIjod4BTRuKNozEdmDxJSMHE4i683R/AW3PCd3VAXcJ+m7DwLHIr1GG3p720fsUEa6qBpH3JEJKAolrFi0X4tjrQH3Djsj4F13ndCdsdxG6DYDz6avEkfFQ+xhtNMr94fvouf9UBxKJEYgMaqrfc7pHDAInCfDw8DAUy50hMHhVbwUAReoJqy7eD0UaBhNPS4xxHzKBqBsuWuVrkYVap0YAspuRg4c4r7bDfeC2Wveg/8uoAY5kAomNx2m4D6O9BznNweG/vPpz/L+3yVGAupmovzvgLkJ1W0Y2B0/jbx7qPWsMATMGVRV6YgSAU3iL0My2zexDZva+2fltZna/mT1kZu8ws71V20gkzhKR2O3/bPF3z4D/mGHwC0d67nXXrkcvLEUEXfNcRPfTwmm8SvxjAB6k8zcA+IVSytcD+HMArz2FNhKJtaG1cka+d/b7axyAErO/SRgRtDMC3rlI7RC1frXuhb0YPWVXYgJmdguA7wLwa7NzA/CtAN41y/IWAN+3ShuJhKKmAtR05SFrv0YK8gtA/HKQn+/u7i58ydjzRPl3dnYW/v23s7ODnZ2deV1cjs9bfdR74B9w0rBZw6qSwC8C+CkArqh8NYC/KKX47oyPALi58sDuNrMHzOyBK1eurNiNRGIRyiiGbAcRA4nK1iL0/HwZe4XnHbPit6DSy9qYgJl9N4DHSykfXKZ8KeVyKeVSKeXSjTfeuGw3EolTQSsqEDgmUtXzJ5MJDg4OTtgIIltA7ccrNtsbIhWCy2n/I7WjB6t4B14G4HvN7JUArgPwVwC8EcD1ZrYzkwZuAfDoCm0kEiFqQTO9bkM91g1AvZ6a58At8+4V4G3KWeTnfQU9UpDb4+3IdVv0SCqJJI1WAFIPM1iaCZRS7gVw76wTLwfwL0opP2Bmvw3g+wG8HcBdAN6zbBuJxDJgfz2Dr7EbkGMCIibgZfz7BO7+4xBj/dS5hw1zGrsEOZSYy+l9KFpMINqlKCqnWEecwOsAvN3M/jWADwF40xraSCSWQmTtr+VRtaAlukeWfN5UxMs7oihD3oCk1q9a37w/fMwqydpfICqlfADAB2bHnwHw4tOoN5EYAq/6p2FUG4K63/x/yF/P18cYDIf6El3jvrTyOjJiMHHNIyKqiCkw8UY2AVYJWL/3FZrFeo4S1M99sSuQVQDekNS/GRBtKhL118EbjtR2Q4qkhGQCiUQnohXc/2srvl5zgnS1wYmVxXJdrXtW8KjM2PuJkEwgkUD9az3selPCZ71b/fxmFtoE1ACpbxkCx6s9SzNRHIGfRzsia/kWkgkknjbQSd/yDmg53aeP1QEv5wTNL/oAOLG6uyrAG4440/CXg9i1yOVUmlAi1g+StNyCvRJDMoHE0wpDcQKs70euRNWvedWOvkmgbj2/rt8hYObSSmMmE334RO0a2v9ljKPJBBLnFjX9P8oX/aKXf2ouQnXh8d6CTOTAsauwtblI1D99kUlfSqohmUDiXKKmFkTn+sJRlD6mnp6+DPU9YljLukuTCSSedqi5B2sEEhFtSx1gkd7dgLzKs4uQ7Q0s6nP9jmgHItXra0yEVYmhXYkVyQQSiRnG6tPqJah5B9hwqIwo+tpQK8in5v+PPBf830IygURiCTAR81t9TvS1twBrcQK1a3o9Ino9r7k0a0gmkHhaIvKx114sUtGcXw7SiEEW6Xd3d+fitxM9gIU3CI+OjrC3t7eQ5lICt1lKWdinUO/BoS8csarCtgf+8OkQEziN7cUSiY1EZIirGeFUr66lMcPQ66rva/6orppxMbqHWunBvBEAAB0wSURBVP6WYXHIMwCkJJA4Z2itiCxK65eH+OfuO3bFsTvOXyH27w7odl8sCfAq7fmcaHveB9Dr0eamQ/edTCBxrjCkJtRWdP/3n3sFOGaA07weVj96JYDIc8DlNV/UR6Df0JlMIHEuwUTDxMJEqvaDSPznFVjdh7x7EHAcTcjtMyNx+0IUk8Dt6zGfex1DdgBGMoFEgjDGTcjqA4v3TvS81yDXHUUDsu7u/8ww2BXJ+VmSUbVAXZg1pGEwkUA9+m9M2ZZPP0qvoeYqjOrUMqoO9AQOpSSQOBdgwqi5DHljEbUTuAuP0/3bA7rSeyThwcEBSinztwhdSvCyntfBLsKarYAZBNsc2EbBdg3f/qy1vVhKAolzjWVW/p46eqL6hjwVY9qutdFTV0oCiUQnWoQbRelF0YQuWbj0wAxEbQJq/GupAdzeZDI58Vn0dBEmEoJWcA7r15HFPrLSswoR7Sfged1oyPsVOjh/TR2oBT9xP3TjkSEkE0icK6hLMLqmrjdlBI7oOjMEdQf6cRRJyPn8n8OCuR22PbDUwPUrUhJIJJbEkIVdxX+HE6iqCE6k0cYhuksRbzzCbTmGxH1WEZIJJBIdqEUT6jv6UdAQr/yRJOAErypCS3SvtRulRxJAjwESSCaQOIdoqQQqVvMKzkyB3X2+gSinu/HP04DF7w4cHBwAWPzWAH+izPvA7x3w242en6MSfXNTjRUYYgQruQjN7Hoze5eZfdLMHjSzl5rZc83s/Wb26dn/Dau0kUhcLYwJveUyutefqgXRtdqv1qchV2AUuFTDqnECbwTwX0spfw3A3wTwIIB7ANxXSrkdwH2z80TimkXk89fzIWKu6ee9IrsSeu1f80f9ViytDpjZcwB8C4AfnDW0D2DfzO4E8PJZtrdg+o3C1y3bTiKxLtQiBzldIw15HwF+9Tf6xLhHGLKqwPsOssvQXYtcPrLys3rA+dg1ONY7sIokcBuAKwB+3cw+ZGa/ZmbPAnBTKeWxWZ7PA7gpKmxmd5vZA2b2wJUrV1boRiKxPHojBmuuN65DXX9aphZnEPVj6Ly3/2zXqGEVJrAD4EUAfrmU8kIAX4aI/mXacth6KeVyKeVSKeXSjTfeuEI3Eon1o0d3VzdhTR1QW0DUTiv6MLqm6WOwChN4BMAjpZT7Z+fvwpQpfMHMLgLA7P/xFdpIJNaC3pU3ihRk8VvVA7fY67Gf69eKOb12rdZe9PPykbRRw9I2gVLK583sc2b2DaWUTwG4A8AnZr+7APzs7P89y7aRSJw1IvHdoZ8HUx2cg4GcoCPbAYCFbxE6eHNTha/w+qYg2yGitnsYwapxAv8UwFvNbA/AZwD8EKbSxTvN7LUAPgvgVSu2kUhcNUQReryxB4vmfo13C4p+HKfg/xq7UBPpI2+E1ldmMQU99QErMoFSyocBXAqS7lil3kTiWoIyCl51ewlxbHtalwYH8fFQuxkxmEjMUAsbZvHbQ385BJgJUPV54OR24y6uAzjhWmQ1I5IWWMTnsty/yAvRQm4qkkiMRMtd1yK6nnJjDHq1umtuyBpSEkgkVoSK/C2iq71ezGDJotWmiv5cVvckWKdhMJF42iFalf1cXXZaxsV5jgb0iEFOi94ijD4+wq8cR6s9eyTGSA6MZAKJcw+1zNfSNR/r+LwiR4TeywRYkuBjbgs4uQUZMwXtzxCSCSQSA+h5uUct8nwcRReOaUevD20jzlGDamSMkEwgkRiAeg38v2V4q1noo3NtR20CtXZ6jJA9KkIygUQCdZFf8+i/fwLcwTq6v0UYRRTyxiHaBw8+YjWC7QLsCoz6yKHDfq1prBwzUIlE4hitrcFqq3/kvmtJB46IMal6Uas3mUAicUYYIvAeYlcXnxL6mKjDdBEmEitA1YOau5A/SwYsWvT1bT5O82g/Jc7Ius+eAI0+5Ov8STQvq3kjJBNIJEaC7QbRjsB+nYm0ZuRj4lSmEJXntFresUgmkEgsiSGCi4jfj2tMIHInDqkAqvtrVGJKAonEKaK2knMQT7TxB5evGetarkBlJkOeDM9b+6QZI5lAItEJ9eP7NSY0ZQAte4EScuT6U5tExCBaHoAe70AygURiBFpEqNdq9oLaMTOFoQjDIaJvSSOKdBEmEhUMGdsiX3xEcBxPoGWj9k6zXz11pySQSMxQs8JHxj3NU5MQok+QA/X4/xbxDon7rT0HI0bkSEkgkZhhme2/aqqAH/P1VvqY9mqouRHTJpBIjESLYFQyiKz2ft7Uw8kIOPbtQmUiUehwZBuoIZlAIjGAlu4+pNdHngElWkeLodTq52vKlFgVSHUgkVgSQ1KB/7fUAj1nN2GrvR7mU1Mtog+Z1JCSQCIxgB5GMCatJwKwpgIwXKWo7XXgkoB/9aiGlAQSiTWgx/jX2iFoWcaj+XpsAskEEokVUHMNRmnL1t1SAxS8l6F+87CGlZiAmf2EmX3czD5mZm8zs+vM7DYzu9/MHjKzd9j0E2WJxDUN1ft7ib03KKimIrQ2Lunpr34PMWyju9aTjdwM4J8BuFRK+UYA2wBeDeANAH6hlPL1AP4cwGuXbSOROEuoe60mRg8F7UT5evz7PZuR9jAV9Qq0GACwujqwA+AZZrYD4JkAHgPwrZh+phwA3gLg+1ZsI5HYWPQyiZ46ehgJn7tRkPclZCaiBsIalmYCpZRHAfwcgD/DlPi/COCDAP6ilHI4y/YIgJuj8mZ2t5k9YGYPXLlyZdluJBJrx6q6/Vmgtt9gS6JxrKIO3ADgTgC3AfgaAM8C8IoRnb5cSrlUSrl04403LtuNROJMMKQeLKsK1PJwfa1VPCJ2VgP4vIZV1IFvA/BwKeVKKeUAwLsBvAzA9TP1AABuAfDoCm0kEtcMVJfXnYFaOwXVyva0GeVlhrBOF+GfAXiJmT3Tpi3cAeATAH4fwPfP8twF4D0rtJFIbBzGWPc13f/1WwWu06uer3lb9beYwVqYQCnlfkwNgP8HwEdndV0G8DoAP2lmDwH4agBvWraNRGJTMWShX/WNRK5PRfmIqDkeIHJlri1suJTyegCvl8ufAfDiVepNJJ4uWPabActCbQPRtuaKfHcgkTgFDOn7NWagx1HMgKsDkTtQPz46mUxOrP5Dm41m2HAicRWxzDsCPcFEUYRjDSkJJBJLYJnPgbW2Ket1L7a+PcifIVe1IJlAInGK6HXnqWg/lDf66Ih+jiz6QAnX6aqDvnOQTCCRWBMi/b6l57fStV52G/K1yG4QRQz2xAgAaRNIJE4VQ1JBhCEijd4LqMUEaJTg2l2EicR5wxhbwFBkYE9dNTVhqB2NE2ghmUAi0Ykxq3xN9Aew4NaLdH2OHJxMJgvqQGQXYPCqz580azGcZAKJxIowa38YVBF9nqxWPor+0zQNK47yZ5xAInEKGOPTV+KrvdYbXYtW91qocG2lV3tAegcSiRXR0uUjl150fTKZzNNYxHeR38X9w8PDeVlP8xX/6OhoXg+rCg4PFW7FJCiSCSQSA2iJ+r0xApFbMLILsEtQ86otIQocGnqxKUIygURizYh0fmUA6gLkPFyPbxzKjCFyA3q+yWSS6kAisQp63Xi145pHoPZr5eN6asZA7cfQJqNAMoFEIsRYt1/LFhC98ccrvxK2ugpZRYg2HQHa7yPUohId6R1IJAK0iCpyvY3ZU3CIKLVNfRmI62hFEGodNSQTSCRGYJlXfx01A2EtL9db29KsFjCk52kTSCRGYCjcN7Le+7G6/lisZxehp0WGwcPDw3n+SH2IIgcdbDTsRTKBRIIwxABaKzcTqF+rEWtPPb19GIo4HEIygUSC0CKolvGvxgBq4cGRLcHdfWzRZ0mA62n1d2y8QDKBRGKGntVXxf/Iqh8Z7aIgIE1T9YG9CVHbzgDYC+ARg46UBBKJDvSqANG/MoKheACuU8uo25BtCDVPQGvFd4lgyD6QTCBx7tGjAkREqAyA6+N/4GRMv1/j1335nFd0r7+3n7U+1JBMIHGu0ZICWipAdFyrM8qnKkStL5EK0TofsjtESCaQOFfoIXrgJJEyserbgJHrjtNYtG9tKNJyKao00FIv+FqPOjAYLGRmbzazx83sY3TtuWb2fjP79Oz/htl1M7NfMrOHzOwjZvaiofoTibNCbYXu/UV19MIt/jXLfbTfQKTf18pp/lF968jzGzj5yfF7ANxXSrkdwH2zcwD4TgC3z353A/jlpXqVSJwiolUz+qlVXq3zLdSImH/+Zp9u9qEbf2haVFfPpiG114sVg+pAKeUPzOxWuXwngJfPjt8C4AOYfoj0TgC/WaYj94dmdr2ZXSylPDbUTiKxDgxZ+Xt1/Zp7r8ZI+NpQ5J/D87FawG3pC0Gs+0d2gF7vwLLvDtxEhP15ADfNjm8G8DnK98js2gmY2d1m9oCZPXDlypUlu5FI1DGWAUQSA1C3D7R+rNer6y9iFNGW4i0DY+0c6JcAHCsbBkspxcxGK0qllMuYfsocly5dWk7RSiQCDFnRI+iqGR1HxjlenaNoP12J2Q3I5b1ez8N11PJxnXq/Khm0sCwT+IKL+WZ2EcDjs+uPAng+5btldi2RWCtahF9bPWv/0bUoCCh6R8DzRgFAUT2sDujXhjgvMxs+HzsuEZZlAu8FcBeAn539v4eu/6iZvR3ANwP4YtoDEutAi8j1uKXrt3R/LqMEPcQEON/h4eG8PmUCQ28WOjQ0WPvq+Zk5tFQLxiATMLO3YWoEfJ6ZPQLg9ZgS/zvN7LUAPgvgVbPsvwvglQAeAvAkgB8aqj+RGIPeFZ8Rid5ah+fRbwJ4ef53eFQf1891RKqF1hu5AY+Ojk6I+FoHH0f9a92vosc78JpK0h1B3gLgR7paTiRGokXwQ+dKqMoYIr2dy/mxthExmBrBax1aLgojVmOfMii9v2WQEYOJjcWYVT8igNa1lqhc8wYMqQP8QhDbBaIIQ+1TTbWoGSqZIURGwzEBRLm9WGIjMXZVWzZarhct4+KyK3BPW61rQzgzF2EicVYYmtQqEkd6dE0dYERuPBXbefVmPV71+cjzEF2vqSARWl4NzdMjESQTSFx19Ez4Mektb0F0HnkGIvG8FuATqQ+1gCDtV6187f6Ygak00pJKVvIOJBKnjWXF3d4JHhFRzRZQ2zQ0ImDPq58L53o0OjAqX+u/EnbrfrRcxJBaY8ZIJpA4M7QmZGSlj/LULO81NWDIPcg+d44a1PweDehp29vbC0THabU+cXsOVjui/kbqSCTi11SfHqRhMHEm6DVsjYl5v5oYY6hb1nDYWs1Pc4xSEkicOlbR8R2tVV/rGVIH+Fzf2mOwa87TIv1e3wpkEVwjAKNQ4Fr/VbWo6f0+BqrKMAPlV42HxjyZQOLU0Zp4Y1aw3gCYMYEyGn4bRQey8a3W51WCc3oRqSQte4oyiV4kE0isBVF03ar1RDq2Xx8z8Tm/2gS0fnf/1RhB72qrfW7dF6/oGhDEaeol4Lr0P12EibViLIGvqiNH563jSB2IrOiRW9CP2SMQqQOsKjj4haCa4bLW35o9ICo3JLWkOpBYC3oIeRliH2NPGMsEmNCB9jcCIveeErrmUyYQ1T8k0teOW+VWNRImE0gshaEXZjQPo0XovRO6Z/JHrja2A0TfAmihFpfPIvvOzjFJ8SvE3geHRge2XIKcr6W2DDGXGpIJJFbGOo1kvRLHWHWA02pW+Ug0j9rqFd9r/asdn4XxEUgmkBAsY2UfYxxbpf0eA5gf11QAvc7nrOvrZp89ej8Quxlb98Ljxv+88nu+6N44TY2G/Fxa45pMILGAsfplj8W7VUbzjakjUgkiA1nNFagxB8rYoj5x21Ga9q2XMUb5ontThlN7XlGfa8iIwcRa0MtMWvmG6mi57VbplzKQobaG2oxUiyithUh10f0Lx9wjIyWBc4Sz0C9726ultXTqWlrNJhBF3tXcgKoeHB4eVgmODX7sKfD6o7o9T7RCe16VPjyN76vGMNS4OAbJBM4JNp0BjLVs12wBQN34p+VqPyXgGlG3wnvH3MsQIuYBnHwZqZW3hWQC5wQ97rTTbm9MndEqNlTHWH27xQCGyqodoFZGVYkhcT2SBKJdhiMbhZep2Q6GbBeOZAJPA4wl4FWDS5bpxzKrvx5Haa3V36/Xzltp+oVg3Tewtvq3JBTgJAMZilVYhjmPjRpMJnANY9nVWyflOvrQQ/StPDWCj4hez6MNP7wc6/DsElSmEH0mnNOivkZ913twEb628nPeZZ7NMmWSCVzD6BG5l7XS9zKYmjGqJt7ztZYhq3Zvq6oIXF6/D8B5vG/8Sq4TP6/ePYyAr9ViKoaYXuQ6jVSFZZBM4BrFaRF/b33LlBsj7vN5SyVotTOk+2sdQ9t9KfS7AKx3R/9aZ01/H7qPWuxE1M4ySCawYTgt3XDMBBlT9zK6vZ4PpdUItaWH1zbucFE+qkP35dNyvCFI5DGo3XeLuUXSUMswqPXWGIxjmeCtwWAhM3uzmT1uZh+ja//WzD5pZh8xs/9kZtdT2r1m9pCZfcrMvmOo/sQxll2NI6yDAawDvQxgTF1KCFtbWwtBP7VoujFRdtpujTFEUklURtN1Y9JIBRiKC6gxB0VPxOBvAHiFXHs/gG8spfwNAH8M4N5ZQy8A8GoAf31W5t+b2XZHG+cWrVWFodFpQ79ejC3TG6XXOm8dqz7ear9Wlgmff9vb23Mdn6/3jGN0rceA17LUs6SiXguHMgPve+05tNSfGnq+RfgHZnarXPtvdPqHAL5/dnwngLeXUp4C8LCZPQTgxQD+12BPrkH0ctpWudY1xmnofquK/bW0MedDev+Q+F87r+0NoJuHqHjPbj/1IkSqhdathr7Wqtxzn35ee94txqPl/HzouZ+GTeCHAbxjdnwzpkzB8cjs2gmY2d0A7gaAr/3arz2FbpwdakTMA94z+LW6avnWaewbQ/S16z16f0Qstbf8/D8Sl1uidGQD0F8UL6CfBm+V8X6PEcsjN2ZrjNRDURu/qJ4xC9RKTMDMfhrAIYC3ji1bSrkM4DIAXLp06eoqpiMREXikV/YwgtNY4XvrbvWlZUSqpbXur5cJ9kKt79ovPmex2AlV//1eIqajq3/tXiKGHzG06JVi7atec1XF7zlSg3hhaDHZoQVkaSZgZj8I4LsB3FGOW3wUwPMp2y2za9c8Wqtcb7noeu3hLMscNnH15+OIWKL02orP1zRd02oWfV3ZWRLwcuo54HJKhBEzifrWUhtahBql1Wwj0XNbiyRgZq8A8FMA/k4p5UlKei+A3zKznwfwNQBuB/C/l2ljU9CacAzVDSMjTYTo4dZWniGxk/u8THrP9V6irxF5dFw71/GrvbATpan4ztc8n4YC10KDtWxksGz1ryX668dL/fmz7ULjE6I5wtei6y0MMgEzexuAlwN4npk9AuD1mHoDLgB4/6yBPyyl/ONSysfN7J0APoGpmvAjpZT69irXCFgUjc45X1R2qO7aeUtK6GEstTw1NaHH2NTbj+hzW1E5raO14utKGqWx+K6MOFrFt7a25q8GswcgYiSR6K5EHjH96BlHTIAJ1u0A29vb4XjWJAxmRL3SZI934DXB5Tc18v8MgJ/pav0aRKT7a3ov8dXSeh7eWHtDrzQyNIGXBU/iaLNMXb28LzWpoJamzMDTOSAnImomKF+Jozp1bFrSUMQwmNFwezpOms/tAxFqY6J9rSEjBgNEK9PQalWz4OoxP1TgJFMZa1AbyttKj9Jakzo6HyP2R/96bWicI7G+hzlEP34RyM+jdjjqMBqPiPiUwUTGPZdEHPpyEZcZInS95mqEthEhmcBIROJe68FEeWvqBKcP9aH3em/escSu+SICjv5baUP/Q64/7UeU7vehDCDK27pXYJxa5dcjSUDHp4ZW3zx9bJQlkEwgRM3oAsQTlEU1ngiR6hCtCC0ss9LXVuNWWg+RD/0PEfoQoxzLaKL7i+pl+LNV9YT7wKoEG+WiOiNJKGIOKgH6Sl1DNFZ8D1Ea2xBYmhhiHhvFBHpWwauF2gqjL6jw66YO1fmG7Ara7tD1FvfvZQI9jCMi9F4mEfnPe8pxO61y2t+efjhqG4Wo3h6pavoMJ5PJAqFzHlYB1Sugbbag+aKxYHVgCBvDBPgBXG1GEE2gKHx0MplgMpngqaeemg/4/v4+ACx8iaYWn17DkMFJz2uE2ZO3VX+0ivj9R+/i90gJNSLXtqOyGnSj0lqtvlbf9L703rn+mvqmDCPqE9cb9evo6Ajb29vdNNAj0akBtoaNYgKHh4fzlzxYnDlrqM6u6oD3b2dnB6UU7O3tzdOf8YxnNANh+KGrRBFd8+PoIxdaP5/36Jq9zILRawjsqW/syh4xk8PDw0HpqjaH1JhbYypDUEmvp66I0UeRhcoUhtQILmdm84WqJS1uFBPwG9QHfbUYAfdNrfd8zZmB59UJqw8hIm5/WJzu2167yqGMgfvHqDGB2ioUndeYQGvVrtlNahgjlQD1kFxnzMq0o/lTC84Z6u9pzMFeI2C0aKhUNmRX4nxDTMCGOnQWMLMrAL4M4Imr3RcAz0P2g5H9WMS13I+/Wkq5US9uBBMAADN7oJRyKfuR/ch+nG0/8jNkicQ5RzKBROKcY5OYwOWr3YEZsh+LyH4s4mnXj42xCSQSiauDTZIEEonEVUAygUTinGMjmICZvcKm3yl4yMzuOaM2n29mv29mnzCzj5vZj82uP9fM3m9mn57933BG/dk2sw+Z2ftm57eZ2f2zMXmHme2dQR+uN7N32fSbEg+a2UuvxniY2U/MnsnHzOxtZnbdWY2Hxd/ZCMfApvilWZ8+YmYvWnM/1vO9D41wO+sfgG0AfwLg6wDsAfi/AF5wBu1eBPCi2fGzMf1+wgsA/BsA98yu3wPgDWc0Dj8J4LcAvG92/k4Ar54d/wqAf3IGfXgLgH80O94DcP1Zjwemu1M/DOAZNA4/eFbjAeBbALwIwMfoWjgGAF4J4L8AMAAvAXD/mvvxdwHszI7fQP14wYxuLgC4bUZP291trXtiddzsSwH8Hp3fC+Deq9CP9wD4dgCfAnBxdu0igE+dQdu3ALgPwLcCeN9sUj1BD3xhjNbUh+fMiM/k+pmOx4wJfA7AczENa38fgO84y/EAcKsQXzgGAP4DgNdE+dbRD0n7ewDeOjteoBkAvwfgpb3tbII64A/dUf1WwbpgZrcCeCGA+wHcVEp5bJb0eQA3nUEXfhHTjVs9wPurAfxFKeVwdn4WY3IbgCsAfn2mlvyamT0LZzwepZRHAfwcgD8D8BiALwL4IM5+PBi1Mbiac/eHMZVCVu7HJjCBqwoz+yoAvwPgx0spX+K0MmWra/Whmtl3A3i8lPLBdbbTgR1Mxc9fLqW8ENN3ORbsM2c0Hjdg+iWr2zDdsfpZOPkZvKuGsxiDIdgK3/uIsAlM4Kp9q8DMdjFlAG8tpbx7dvkLZnZxln4RwONr7sbLAHyvmf0pgLdjqhK8EcD1ZuZveZ7FmDwC4JFSyv2z83dhyhTOejy+DcDDpZQrpZQDAO/GdIzOejwYtTE487lrx9/7+IEZQ1q5H5vABP4IwO0z6+8eph80fe+6G7Xp+5dvAvBgKeXnKem9AO6aHd+Fqa1gbSil3FtKuaWUcium9/4/Sik/AOD3cfyNx7Pox+cBfM7MvmF26Q5Mt44/0/HAVA14iZk9c/aMvB9nOh6C2hi8F8A/nHkJXgLgi6Q2nDrs+Hsf31tOfu/j1WZ2wcxuw9jvfazTyDPCAPJKTK3zfwLgp8+ozb+NqVj3EQAfnv1eiak+fh+ATwP47wCee4bj8HIcewe+bvYgHwLw2wAunEH7fwvAA7Mx+c8Abrga4wHgXwH4JICPAfiPmFq9z2Q8ALwNU1vEAabS0WtrY4CpAfffzebtRwFcWnM/HsJU9/f5+iuU/6dn/fgUgO8c01aGDScS5xyboA4kEomriGQCicQ5RzKBROKcI5lAInHOkUwgkTjnSCaQSJxzJBNIJM45/j/Tc5iV32+TXwAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}