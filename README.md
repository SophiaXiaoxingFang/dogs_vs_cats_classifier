# Dogs_vs_Cats
The project is to implement a binary image classifier to tell a picture is Dog or Cat.
A basic Convolutional Neural Network (6-layer) is built to solve the problem using `TFlearn` (http://tflearn.org/) (a higher-level API for `TensorFlow`).

The data-set has been obtained from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats), which is stored in the **./data/** folder.

The function of each Python file is explained as follows:
1. **preprocessing.py** Pre-process the data-set, including the training data (**./data/train/**) and the testing data(**./data/test/**). For decreasing the complexity of computation, each image is converted to the size of 100*100 and gray-scale. After been shuffled, the data will been stored in two files: `train_data.npy` and `train_data.npy`. 
2. **cnn_model.py** The 6-layer CNN module in which the learning rate is set to 1e-3.
3. **training.py** The script to build , train and test the CNN image classifier. 
    * During the training procedure, 500 samples of the training data-set are assigned for validation. The training procedure is set to take 5 epochs.
    * When the training is finished, 20 images from the testing data-set will be shown with the prediction result of the trained classifier. 
    * When the figure is closed, the `submission_file.csv` will be generated with all the prediction results (Dog '1' or Cat '0'') of the images of the **./data/test/** folder.
4. **get_image_classification_folder.py** (Optional) Generate two separated folders (**./result/dogs/**, **./result/cats/**) and copy the classified images to these folders accordingly.


## Reference
https://www.geeksforgeeks.org/image-classifier-using-cnn/
