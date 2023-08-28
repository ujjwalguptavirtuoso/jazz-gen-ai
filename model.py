from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_shape), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
