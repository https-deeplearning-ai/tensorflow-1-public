
import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],dtype=float)
ys = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],dtype=float)

model.fit(xs,ys,epochs=500)

print("x:11.0, prediction y:",model.predict([11.0]))
print("x:6.0.0, prediction y:",model.predict([6.0]))
