from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mas_lib.preprocessing import simplepreprocessor
from mas_lib.datasets import simpledatasetloader
from imutils import paths

# paths to image dataset
PATH = ""
IMAGE_PATH = list(paths.list_images(PATH))

# initialize the dataset loader and preprocessors
sp = simplepreprocessor.SimplePreprocessor(32, 32)
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp,])

# loading the images and labels from disk and
# flatten the imaages
(data, labels) = sdl.load(IMAGE_PATH, verbose=500)
data = data.reshape((-1, 3072))

# encoding labels for computation
le = LabelEncoder()
labels = le.fit_transform(labels)

# split the dataset into 75% training and 25% testing
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

# loop over different types of regularizers
# to see the best performing
for r in (None, "l1", "l2"):
    print(f"[INFO]: Training SGD classifier with {r} regularization")
    model = SGDClassifier(loss="log_loss", penalty=r, max_iter=10, learning_rate="constant", tol=1e-3, eta0=0.01, random_state=42)
    
    # training the classifier
    model.fit(x_train, y_train)
    # evaluating the classifier
    accuracy = model.score(x_test, y_test)

    print(f"[INFO]: Regularizer: {r}, Accuracy: {int(accuracy * 100)}%")