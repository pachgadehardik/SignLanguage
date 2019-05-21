import keras
import numpy as np
from keras.models import load_model
import videoto3d

model = load_model('d_3dcnnmodel-36-0.97.hd5')
img_rows, img_cols, frames = 32, 32, 10
channel = 3
vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
X = vid3d.video3d('train/1/001_001_001_right.avi', color=False, skip=True)
# X = np.array(X).transpose((0, 2, 3, 1))
X = np.expand_dims(X , axis=0)
X = X.reshape((1,10,32,32,1))

X = X.astype('float32')

# model.summary()
print(X.shape)
y_pred = model.predict(X)
print(y_pred)