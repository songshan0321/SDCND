import os, csv, cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math

### Load dataset from csv ####
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

# with open('./recover_data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader, None)
#     for line in reader:
#         samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

def flip_data(image, angle):
    image_flipped = np.fliplr(image)
    angle_flipped = -angle
    return image_flipped, angle_flipped

# def random_shift(image, angle):
#     h,w,_ = image.shape
#     # allow shift up to px pixels in x and y directions
#     px = 40
#     x = np.random.randint(-px,px)
#     T = np.float32([[1, 0, x], [0, 1, 0]]) 
#     image_shifted = cv2.warpAffine(image, T, (w, h)) 
#     angle_shifted = 0.02 * px
#     return image_shifted, angle_shifted
    

def generator(samples, batch_size=32, train_mode=False):
    print("Number of original data: {0}".format(len(samples)))
    # Convert into images
    images = []
    angles = []
    for sample in samples:
        center_angle = float(sample[3])
        if train_mode and abs(center_angle) < 0.0001:
            continue
#             num = np.random.randint(0,100)
#             if num >= 30:
#                 continue
                
        filename = sample[0].split('/')[-1]
        current_path = os.path.join('./data/IMG/', filename)
        center_image = cv2.imread(current_path)
        
        # Pre-processing
#         print(center_image.shape)
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
#         center_image = center_image[:,:,np.newaxis]
#         print(center_image.shape)
#         if abs(center_angle) < 0.01 and augmentation:
#             center_image, center_angle = random_shift(center_image, center_angle)
        
        images.append(center_image)
        angles.append(center_angle)
        
        if train_mode:
            image_flipped, angle_flipped = flip_data(center_image, center_angle)
            images.append(image_flipped)
            angles.append(angle_flipped)
        
    X = np.array(images)
    y = np.array(angles)
    
    num_samples = len(X)
    if train_mode:
        print("Number of augmented data: {0}".format(num_samples))
    while 1:
        sklearn.utils.shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]
            yield sklearn.utils.shuffle(X_batch, y_batch)

#### Model ####

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 128

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((80, 25),(0, 0))))
model.add(Convolution2D(24, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=1, activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=1, activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=1, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

#### Training
train_generator = generator(train_samples, train_mode=True)
valid_generator = generator(valid_samples, train_mode=False)

model.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
model_checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
history_object = model.fit_generator(train_generator, \
                                    steps_per_epoch= math.ceil(len(train_samples)/batch_size), \
                                    validation_data=valid_generator, \
                                    validation_steps= math.ceil(len(valid_samples)/batch_size), \
                                    epochs=20, verbose=1, \
                                    callbacks=[early_stopping, model_checkpoint])

# model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.yscale("log")
plt.savefig('loss.png')

    