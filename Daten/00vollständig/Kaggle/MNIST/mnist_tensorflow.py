import logging

DATASET_FILE_PATH = "./source/mnist-original.mat"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('dev.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

logger.info("스크립트 시작")

logger.info("필수 라이브러리 불러오는 중...")
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Flatten,Dense
logger.info("필수 라이브러리 불러옴")


logger.info("Tensorflow version: ", tf.__version__)

# 데이터셋 불러오기
mnist = loadmat(DATASET_FILE_PATH)
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]

logger.info("Dataset imported", mnist)
logger.info("Dataset Check data: ",mnist_data)
logger.info("Dataset Check label: ",mnist_label)

    
# 데이터셋 로드 및 스플릿
logger.info("Ready for Split data for training")
X = mnist_data
y = mnist_label

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2024)

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

logger.info("Data for training Splited")


logger.info("Normalization Start")
X_train, X_test = X_train / 255.0 , X_test / 255.0
logger.info("Normalization End")    
    

logger.info("Dimension add Start")
X_train = X_train[..., tf.newaxis].astype("float32")
X_test = X_test[..., tf.newaxis].astype("float32")
logger.info("Dimension add End")


logger.info("Make batch Start")
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
logger.info("Make batch End")

logger.info("Initializing Model Class")

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu', padding="same")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
logger.info("Model Class Initialized")


logger.info("Creating Model instance")
model = MyModel()
logger.info("Model instance created")

logger.info("loss function & optimizer setting")
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
logger.info("loss function & optimizer setting done")

logger.info("metrics setting")
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
logger.info("metrics setting done")

logger.info("model train function setting start")
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
logger.info("model train function setting done")

logger.info("model test function setting start")
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
logger.info("model test function setting done")

EPOCHS = 5

try : #model train and test and show metrics ( main )
    logger.info(f"model main start, Epoch : {EPOCHS}")
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
    logger.info("model main end")
    
except Exception as e: 
    logger.error("Model main run failed",str(e), exc_info=True)

logger.info("스크립트 종료")
