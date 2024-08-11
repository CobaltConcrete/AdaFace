import net
import torch
import os
from face_alignment import align
import numpy as np
from torch.nn.functional import cosine_similarity
import numpy as np
from numpy.linalg import norm
import time

ARCHITECTURE = 'ir_101'

adaface_models = {
    'ir_101':"pretrained/adaface_ir101_webface12m.ckpt",
    'ir_50':"pretrained/adaface_ir50_webface4m.ckpt"
}

def load_pretrained_model(architecture=ARCHITECTURE):    
    # load model and pretrained statedict  
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

def load_pretrained_model_w_cuda(architecture=ARCHITECTURE):
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # load model and pretrained statedict  
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            model.cuda()
            model.eval() # running this into eval mode 
            return model

def to_input_w_cuda(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor.cuda()

if __name__ == '__main__':

    model = load_pretrained_model(architecture=ARCHITECTURE)
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = 'face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img, bbox = align.get_aligned_face(path, source='database')
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, x = model(bgr_tensor_input)
        # print(len(feature[0]))
        features.append(feature)

    # for feature in features:
    #     print(feature.detach().numpy()[0])
    #     print(len(feature.detach().numpy()[0]))

    print(len(features))
    print(len(features[0]))
    print(len(features[0][0]))
    similarity_scores = torch.cat(features) @ torch.cat(features).T
    
    print(similarity_scores)
    print(similarity_scores[0])
    nested_list = similarity_scores.tolist()
    print(nested_list)
    new_list = [sublist[2] for sublist in nested_list]
    print(new_list)

    

    
    

