"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import pandas as pd
import argparse
from PIL import Image
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images, model_selection


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        out_of_range = True
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                out_of_range = False
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x, out_of_range

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x, out_of_range = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x, out_of_range


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output, out_of_range = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam, out_of_range

parser = argparse.ArgumentParser(description='Layer activation with guided backpropagation')
parser.add_argument('--data_root_dir', type = str, default = '/home/mahmood1/matteo/pytorch-cnn-visualizations/kidney_test_images')
parser.add_argument('--model', type = str, choices = ['AlexNet', 'vgg', 'resnet50', 'custom_resnet'], default = 'resnet50')
parser.add_argument('--model_path', type = str, default = '/media/mahmood1/369612f6-9b22-4858-ba4b-aa97f2e1fb2e/model_testing_results/VGG_checkpoint_epoch_2.ckpt')
parser.add_argument('--n_classes', type = int, default = 3)
parser.add_argument('--k_start', type= int, default=0)
parser.add_argument('--k_stop', type = int, default = 1)
args = parser.parse_args()

path = args.data_root_dir

if __name__ == '__main__':
    
    pretrained_model, model_name = model_selection(args.model, args.n_classes, args.model_path)
    print('Using model ' + model_name)
    failure_dset = pd.DataFrame({'class' : [], 'image_name' : [], 'layer' : []})
    out_of_range = False

    for cnn_layer in range(args.k_start, args.k_stop):
        print('cnn_layer = ' + str(cnn_layer))
        for target_class in os.listdir(path):
            class_path = os.path.join(path, target_class)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)

                (original_image, prep_img, image_name) = get_example_params(img_path)
                file_name_to_export =  'layer_' + str(cnn_layer) + '_class_' + str(target_class) + '_' + image_name
                
                # Grad cam
                # grad_cam = GradCam(pretrained_model, target_layer=cnn_layer)
                # # Generate cam mask
                # cam, out_of_range = grad_cam.generate_cam(prep_img, int(target_class))
                # if out_of_range:
                #     print('The requested layer value is out of range for the selected model !')
                #     break
                # # Save mask
                # save_class_activation_images(original_image, cam, file_name_to_export)

                try:
                    # Grad cam
                    grad_cam = GradCam(pretrained_model, target_layer=cnn_layer)
                    # Generate cam mask
                    cam, out_of_range = grad_cam.generate_cam(prep_img, int(target_class))
                    if out_of_range:
                        print('The requested layer value is out of range for the selected model !')
                        break
                    # Save mask
                    save_class_activation_images(original_image, cam, file_name_to_export)
                
                except :
                    a_row = pd.DataFrame({'class' : [target_class], 'image_name' : [image_name], 'layer' : [cnn_layer]})
                    failure_dset = pd.concat([failure_dset, a_row], ignore_index = True)
                    print('Failed to generate output for image ' + image_name + ' at layer ' + str(cnn_layer))

    failure_dset_path = os.path.abspath(os.path.join(args.data_root_dir, os.pardir))
    failure_dset_path = os.path.join(failure_dset_path, 'results', 'gradcam_failure_dset.csv')
    failure_dset.to_csv(failure_dset_path, index = False)