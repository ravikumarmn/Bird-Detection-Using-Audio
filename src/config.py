EXPERIMENT_NAME = "bird_name_classification"
MAPPING = {
    0: "American Robin",
    1: "Bewick's Wren",
    2: "Northern Cardinal",
    3: "Northern Mockingbird",
    4: "Song Sparrow"
}
BATCH_SIZE = 32#32
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_CLASSES = 5
DEVICE = "cpu"
IMAGE_TRANSFORM = True
PATIENCE = 5
DEBUG = False
DEBUG_BATCH = 10
DROPOUT = 0.2
L2_LAMBDA = 0.01
NUM_FOLDS = 5
SAVE_DIR_PATH = f"checkpoints/{EXPERIMENT_NAME}_bs_{BATCH_SIZE}_lr_{LEARNING_RATE}_epochs_{EPOCHS}_device_{DEVICE}_droupout_{DROPOUT}_regularization_{L2_LAMBDA}_fold_{NUM_FOLDS}.pt"
