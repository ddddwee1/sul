import layers2 as L
import modeleag as M
import tensorflow as tf

class network(M.Model):
    def initialize(self):
        self.c1 = M.ConvLayer(7, 16, stride=1, activation=M.PARAM_RELU)
        self.p1 = M.maxPool(2)
        self.c2 = M.ConvLayer(5, 32, stride=1, activation=M.PARAM_RELU)
        self.p2 = M.maxPool(2)
        self.c3 = M.ConvLayer(5, 64, stride=1, activation=M.PARAM_RELU)
        self.p3 = M.maxPool(2)
        self.c4 = M.ConvLayer(5, 64, stride=1, activation=M.PARAM_RELU)
        self.p4 = M.maxPool(1)
        self.c5 = M.ConvLayer(3, 128, stride=1, activation=M.PARAM_RELU)
        self.p5 = M.maxPool(1)

        self.fc1 = M.Dense(100, activation=M.PARAM_RELU)
        self.fc2 = M.Dense(500, activation=M.PARAM_RELU)
        self.fc3 = M.Dense(1)

    def forward(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.p3(x)
        x = self.c4(x)
        x = self.p4(x)
        x = self.c5(x)
        x = self.p5(x)
        x = tf.image.resize_images(x, (4,16))
        x = M.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


