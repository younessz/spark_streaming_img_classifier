from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from numpy import savez
from os.path import join
########################################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
########################################################################
x_train[:5]
########################################################################
y_train[:5]
########################################################################
# each image is 28X28 pixels - RGB color coded
x_train.shape
########################################################################
# split the train set into train and validation sets
x_train, x_validation, y_train, y_validation = train_test_split(

    x_train, y_train, test_size=0.15,
    # ensuring reproducible results
    random_state=42)
########################################################################
# flatten images (1 image spread across n columns)
cols = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(-1, cols)

x_test = x_test.reshape(-1, cols)

x_validation = x_validation.reshape(-1, cols)
########################################################################
# standardize the range of RGB values from [0, 255] => [0, 1]
# to speed up the fitting time
x_train = x_train / 255
x_test = x_test / 255
x_validation = x_validation / 255
########################################################################
# saving data

savez(
    # ensure interoperability across different OSs
    join('app', 'data', 'modeling_data.npz'),

    x_train=x_train, y_train=y_train,
    x_test=x_test, y_test=y_test,
    x_validation=x_validation, y_validation=y_validation
)
