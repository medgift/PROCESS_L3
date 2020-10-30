from keras import backend as K 
from keras.layers import Conv2D
from keras.models import Model
import cv2
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_cam_model_resnet101(model, last_conv_layer=None, pred_layer=None):
    n_classes = model.output_shape[-1]
    final_params = model.get_layer(pred_layer).get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, n_classes), final_params[1])

    last_conv_output = model.get_layer(last_conv_layer).output
    # upgrade keras to 2.2.3 in order to use UpSampling2D with bilinear interpolation
    # x = UpSampling2D(size=(32, 32))(last_conv_output)
    x = Conv2D(filters=n_classes, kernel_size=(1, 1), name='predictions_2')(last_conv_output)

    cam_model = Model(inputs=model.input, outputs=[model.output, x])
    cam_model.get_layer('predictions_2').set_weights(final_params)
    return cam_model

def postprocess(preds, cams, top_k=1):
    idxes = np.argsort(preds[0])[-top_k:]
    class_activation_map = np.zeros_like(cams[0, :, :, 0])
    for i in idxes:
        class_activation_map += cams[0, :, :, i]
    return class_activation_map

def get_cam_model_iv3(model, last_conv_layer=None, pred_layer=None):
    n_classes = model.output_shape[-1]
    final_params = model.get_layer(pred_layer).get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, n_classes), final_params[1])

    last_conv_output = model.get_layer(last_conv).output
    # upgrade keras to 2.2.3 in order to use UpSampling2D with bilinear interpolation
    # x = UpSampling2D(size=(32, 32))(last_conv_output)
    x = Conv2D(filters=2048, kernel_size=(1, 1), name='extra_layer1_2')(last_conv_output)
    x = Conv2D(filters=512, kernel_size=(1,1), name='extra_layer2_2')(x)
    x = Conv2D(filters=256, kernel_size=(1,1), name='extra_layer3_2')(x)
    x = Conv2D(filters=n_classes, kernel_size=(1,1), name='predictions_2')(x)
    cam_model = Model(inputs=model.input, outputs=[model.output, x])

    final_params = model.get_layer('finetuned_features1').get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, 2048), final_params[1])
    
    cam_model.get_layer('extra_layer1_2').set_weights(final_params)
    final_params = model.get_layer('finetuned_features2').get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, 512), final_params[1])
    cam_model.get_layer('extra_layer2_2').set_weights(final_params)
    final_params = model.get_layer('finetuned_features3').get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, 256), final_params[1])
    cam_model.get_layer('extra_layer3_2').set_weights(final_params)
    final_params = model.get_layer(pred_layer).get_weights()
    final_params = (final_params[0].reshape(1, 1, -1, n_classes), final_params[1])
    cam_model.get_layer('predictions_2').set_weights(final_params)
    
    return cam_model
def cam(model, img, last_conv, original_size, pred_layer=None):
    if pred_layer is None:
        pred_layer = model.layers[-1].name
    cam_model = get_cam_model_resnet101(model, last_conv_layer=last_conv, pred_layer=pred_layer)
    preds, cams = cam_model.predict(img)
    class_activation_map = postprocess(preds, cams)
    return cv2.resize(class_activation_map, original_size)