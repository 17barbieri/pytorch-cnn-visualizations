"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import os
import pdb
import argparse
import pandas as pd
from torch.nn import ReLU

from new_misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency,
                            model_selection)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = list(self.model._modules.items())[0][1]
        try:
            if len(first_layer)>1:
                first_layer = first_layer[0]
        except:
            pass
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, cnn_layer, filter_pos):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        out_of_range = True
        idx=0
        for _, (layer_name, layer) in enumerate(self.model._modules.items()):
            try:
                for i in range(len(layer)):
                    sub_layer = layer[i]
                    x = sub_layer(x)
                    if (i+idx) == cnn_layer:
                        # (forward hook function triggered)
                        out_of_range = False
                        break
                if out_of_range == False:
                    break
                idx+=i
            except:
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if idx == cnn_layer:
                    # (forward hook function triggered)
                    out_of_range = False
                    break
                idx+=i

        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr, out_of_range

parser = argparse.ArgumentParser(description='Layer activation with guided backpropagation')
parser.add_argument('--data_root_dir', type = str, default = '/home/mahmood1/matteo/pytorch-cnn-visualizations/kidney_test_images')
parser.add_argument('--model', type = str, choices = ['AlexNet', 'vgg', 'resnet50', 'custom_resnet'], default = 'vgg')
parser.add_argument('--model_path', type = str, default = '/media/mahmood1/369612f6-9b22-4858-ba4b-aa97f2e1fb2e/model_testing_results/VGG_checkpoint_epoch_2.ckpt')
parser.add_argument('--n_classes', type = int, default = 3)
parser.add_argument('--k_start', type= int, default=0)
parser.add_argument('--k_stop', type = int, default = 1)


args = parser.parse_args()

if __name__ == '__main__':
    pretrained_model, model_name = model_selection(args.model, args.n_classes, args.model_path)
    print('Using model ' + model_name)
    failure_dset = pd.DataFrame({'class' : [], 'image_name' : [], 'layer' : [], 'filter_pos' : []})
    out_of_range = False

    filter_pos = 5

    path = args.data_root_dir

    k_start = args.k_start
    k_stop = args.k_stop if (args.k_stop > k_start) else k_start + 1
    # if args.k_stop <=k_start:
    #     k_stop = k_start + 1
    # else:
    #     k_stop = args.k_stop
    # print(k_start, k_stop)

    for cnn_layer in range(k_start, k_stop):
        if out_of_range:
            break

        print('cnn_layer = ' + str(cnn_layer))

        for target_class in os.listdir(path):
            if out_of_range:
                break
            class_path = os.path.join(path, target_class)
            for img in os.listdir(class_path):

                img_path = os.path.join(class_path, img)
                (original_image, prep_img, image_name) = get_example_params(img_path, args.n_classes)

                # File export name
                file_name_to_export =  'layer_' + str(cnn_layer) + '_class_' + str(target_class) + '_' + image_name + '_filter' + str(filter_pos)
                
                # File export name
                file_name_to_export =  'layer_' + str(cnn_layer) + '_class_' + str(target_class) + '_' + image_name + '_filter' + str(filter_pos)
                # Guided backprop
                GBP = GuidedBackprop(pretrained_model)
                # Get gradients
                guided_grads, out_of_range = GBP.generate_gradients(prep_img, target_class, cnn_layer, filter_pos)
                if out_of_range:
                    print('The requested layer value is out of range for the selected model !')
                    break
                # Save colored gradients
                save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
                # Convert to grayscale
                grayscale_guided_grads = convert_to_grayscale(guided_grads)
                # Save grayscale gradients
                save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
                # Positive and negative saliency maps
                pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
                save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
                save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')

    #             try:
    #                 # File export name
    #                 file_name_to_export =  'layer_' + str(cnn_layer) + '_class_' + str(target_class) + '_' + image_name + '_filter' + str(filter_pos)
    #                 # Guided backprop
    #                 GBP = GuidedBackprop(pretrained_model)
    #                 # Get gradients
    #                 guided_grads, out_of_range = GBP.generate_gradients(prep_img, target_class, cnn_layer, filter_pos)
    #                 if out_of_range:
    #                     print('The requested layer value is out of range for the selected model !')
    #                     break
    #                 # Save colored gradients
    #                 save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    #                 # Convert to grayscale
    #                 grayscale_guided_grads = convert_to_grayscale(guided_grads)
    #                 # Save grayscale gradients
    #                 save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    #                 # Positive and negative saliency maps
    #                 pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    #                 save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    #                 save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')

    #             except :
    #                 a_row = pd.DataFrame({'class' : [target_class], 'image_name' : [image_name], 'layer' : [cnn_layer], 'filer_pos' : [filter_pos]})
    #                 failure_dset = pd.concat([failure_dset, a_row], ignore_index = True)
    #                 print('Failed to generate output for image ' + image_name + ' at layer ' + str(cnn_layer))
    # failure_dset_path = os.path.abspath(os.path.join(args.data_root_dir, os.pardir))
    # failure_dset_path = os.path.join(failure_dset_path, 'results', 'layer_activation_failure_dset.csv')
    # failure_dset.to_csv(failure_dset_path, index = False)