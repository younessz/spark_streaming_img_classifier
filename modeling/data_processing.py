from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from numpy import unique, savez, prod
from os.path import join
from numpy import newaxis
from tensorflow.image import grayscale_to_rgb
from tensorflow import convert_to_tensor
########################################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# handwritten single digits between 0 and 9 (y target variables)
########################################################################
x_train[:5]
########################################################################
y_train[:5]
########################################################################
# each image is a 28X28 pixels - grayscale color coded
x_train.shape
########################################################################
# set images as 3D arrays (i.e. add 1 explicit grayscale channel)
x_train = x_train[..., newaxis]

x_test = x_test[..., newaxis]
########################################################################
x_train = convert_to_tensor(x_train)

x_test = convert_to_tensor(x_test)

# convert images to RGB format
x_train = grayscale_to_rgb(x_train)

x_test = grayscale_to_rgb(x_test)

# set th Tensor objects back as 3D arrays (with RGB channels)
x_train = x_train.numpy()

x_test = x_test.numpy()
########################################################################
# split the train set into train and validation sets
x_train, x_validation, y_train, y_validation = train_test_split(

    x_train, y_train, test_size=0.15,
    # ensuring reproducible results
    random_state=42)
########################################################################
# ensuring all categories are present among the target variables
unique(y_train)
unique(y_test)
unique(y_validation)
########################################################################
# flatten images (1 image spread across n columns)
cols = prod(x_train.shape[1:])  # cols = axis 1 * axis 2 * axis 3

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
    join('Data', 'modeling_data.npz'),

    x_train=x_train, y_train=y_train,
    x_test=x_test, y_test=y_test,
    x_validation=x_validation, y_validation=y_validation
)
