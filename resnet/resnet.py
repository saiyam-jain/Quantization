import tensorflow as tf


class LambdaLayer:
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def __call__(self, x):
        return self.lambd(x)


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            tf.pad(x[:, :, ::2, ::2], [[0, 0], [planes//4, planes//4], [0, 0], [0, 0]], "constant", 0))
            elif option == 'B':
                self.shortcut = tf.keras.Sequential(
                     tf.keras.layers.Conv2D(self.expansion * planes, kernel_size=1, strides=stride, use_bias=False),
                     tf.keras.layers.BatchNormalization()
                )

    def __call__(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # self.in_planes = 16

        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = tf.keras.layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        sequential = tf.keras.Sequential()
        for stride in strides:
            layers.append(block(planes, stride))
            # self.in_planes = planes * block.expansion

        return sequential(*layers)

    def __call__(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = tf.keras.layers.AveragePooling2D(out.size()[3])(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()