import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)


command_args = parser.parse_args()


describles = {}
describles['glioma'] = "This is an MRI scan of a brain with a glioma tumor, which should be identified accurately."
describles['meningioma'] = "This is an MRI scan of a brain with a meningioma tumor, which should be identified accurately."
describles['pituitary'] = "This is an MRI scan of a brain with a pituitary tumor, which should be identified accurately."
describles['no_tumor'] = "This is an MRI scan of a brain without any tumors, which should be identified accurately."



FEW_SHOT = command_args.few_shot 

# init the model
# replace with my trained checkpoint when complete 
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_visa/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

#initliaize model from checkpoints
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

print(f'[!] init the 7b model over ...')

"""Override Chatbot.postprocess"""
p_auc_list = []
i_auc_list = []

def predict(
    input, 
    image_path, 
    normal_img_path, 
    max_length, 
    top_p, 
    temperature,
    history,
    modality_cache,  
):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })

    return response, pixel_output

input = "Is there any anomaly in the image?"
root_dir = '../data/brain_data'

mask_transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])

CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

precision = []

for c_name in CLASS_NAMES:
    normal_img_paths = ["../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4).zfill(3)+".png", "../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + 1).zfill(3)+".png",
                        "../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + 2).zfill(3)+".png", "../data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + 3).zfill(3)+".png"]
    normal_img_paths = normal_img_paths[:command_args.k_shot]
    right = 0
    wrong = 0
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                if FEW_SHOT:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, normal_img_paths, 512, 0.1, 1.0, [], [])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input, file_path, [], 512, 0.1, 1.0, [], [])
                is_normal = 'good' in file_path.split('/')[-2]

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                
                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

                p_label.append(img_mask)
                p_pred.append(anomaly_map)

                i_label.append(1 if not is_normal else 0)
                i_pred.append(anomaly_map.max())

                position = []

                if 'good' not in file_path and 'Yes' in resp:
                    right += 1
                elif 'good' in file_path and 'No' in resp:
                    right += 1
                else:
                    wrong += 1

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100,2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100,2)
    
    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * right / (right + wrong))

    print(c_name, 'right:',right,'wrong:',wrong)
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

print("i_AUROC:",torch.tensor(i_auc_list).mean())
print("p_AUROC:",torch.tensor(p_auc_list).mean())
print("precision:",torch.tensor(precision).mean())