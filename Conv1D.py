from keras.layers import Dense,Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
def create_Conv1D(num_classes, hidden, input):
    model = Sequential()
    model.add(Conv1D(hidden, 1, activation="tanh", input_shape=(input,1)))
    #model.add(Conv1D(128, 1, strides=2, activation="relu"))
    #model.add(Conv1D(256, 1, strides=2, activation="relu"))
    #model.add(Conv1D(512, 1, strides=2, activation="relu"))
    #model.add(Conv1D(1024, 1, strides=2, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(2,name='logits'))
    return model


