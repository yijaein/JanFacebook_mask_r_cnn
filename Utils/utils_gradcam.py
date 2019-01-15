import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.autograd import Variable


class BaseExtractor():
    """
    Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from inter-mediate targeted layers.
    3. Gradients from inter-mediate targeted layers.
    """

    def __init__(self, features, classifier, target_layers):
        self.gradients = []
        self.features = features
        self.classifier = classifier
        self.target_layers = target_layers

    def extract_feature(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        raise NotImplementedError()


class ResNetExtractor(BaseExtractor):
    def __init__(self, model):
        self.model = model
        features = torch.nn.Sequential(*list(model.children())[:-1])
        classifier = model.fc
        target_layers = ['7']

        super(ResNetExtractor, self).__init__(features, classifier, target_layers)

    def __call__(self, x):
        # extract feature
        target_activations, output = self.extract_feature(x)
        output = output.view(output.size(0), -1)

        output = self.classifier(output)
        return target_activations, output


class DenseNetExtractor(BaseExtractor):
    def __init__(self, model):
        self.model = model
        features = model.features
        classifier = model.classifier
        target_layers = ['norm5']

        super(DenseNetExtractor, self).__init__(features, classifier, target_layers)

    def __call__(self, x):
        # extract feature
        target_activations, output = self.extract_feature(x)
        output = F.relu(output, inplace=True)

        output = F.avg_pool2d(output, kernel_size=self.model.avg_pooling_size, stride=1).view(output.size(0), -1)

        output = self.classifier(output)
        return target_activations, output


class GradCam():
    def __init__(self, extractor, use_cuda):
        self.cuda = use_cuda
        self.extractor = extractor

    def __call__(self, input, index=None):
        input = input.cuda() if self.cuda else input
        features, output = self.extractor(input)

        pred = np.argmax(output.cpu().data.numpy())
        index = pred if index == None else index

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = one_hot.cuda() if self.cuda else one_hot
        one_hot = torch.sum(one_hot * output)

        self.extractor.features.zero_grad()
        self.extractor.classifier.zero_grad()
        one_hot.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        return cam, pred


def blend_transparent(face_img, overlay_t_img, transparency):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    overlay_mask *= transparency

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def heat_map_overlay(background_image, heat, transparency=0.7, blur_times=1):
    cmap = plt.get_cmap('jet')
    cmap._init()

    alphas = np.ones(cmap.N)
    alphas[:64] = np.abs(np.linspace(0.0, 0.3, 64))
    alphas[64:192] = np.abs(np.linspace(0.3, 0.9, 128))
    alphas[192:256] = np.abs(np.linspace(0.9, 1.0, 64))
    cmap._lut[:-3, -1] = alphas

    kernel = np.ones((5, 5), np.float32) / 25.0

    for i in range(blur_times):
        heat = cv2.filter2D(heat, -1, kernel)

    heat = heat * 0.7 + 255 * 0.2

    rgba_img = cmap(heat.astype('uint8'))
    rgba_img *= 255.0

    result = blend_transparent(background_image.astype('float32'), rgba_img.astype('float32'), transparency)
    return result


def make_cam_with_image(img, mask, transparency, blur_times):
    img, mask = np.uint8(255 * img), np.uint8(255 * mask)
    result_img = heat_map_overlay(img, mask, transparency, blur_times)
    return result_img

