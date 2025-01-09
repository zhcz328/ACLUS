import os
import cv2
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from pytorch_grad_cam.utils.drawing import Drawer
from pytorch_grad_cam.grad_cam import GradCAM


model_pth_path = './model.pth'
images_dir = './images'
outputs_folder = './cam_ouputs'

if model_pth_path is not None:
    model = torchvision.models.resnet18(pretrained=False)

    state_dict = torch.load(model_pth_path)

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    state_dict = {k: state_dict[k] for k in list(state_dict.keys()) if
                  not (k.startswith('l') or k.startswith('fc'))}
    state_dict = {k: state_dict[k] for k in list(state_dict.keys()) if not k.startswith('classifier')}

    con_layer_names = list(state_dict.keys())
    target_layer_names = list(model.state_dict().keys())
    new_dict = {target_layer_names[i]: state_dict[con_layer_names[i]] for i in range(len(con_layer_names))}

    model_dict = model.state_dict()
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
else:
    model = torchvision.models.resnet18(pretrained=True)

model.cuda()
model.eval()

target_layers = [model.layer4[1].conv2]  # choose the last conv layer of resnet-18

grad_cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=None)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for file in tqdm(os.listdir(images_dir)):
    image_path = os.path.join(images_dir, file)
    image = Image.open(image_path).convert("RGB")
    image = transforms(image)
    image = image.unsqueeze(0)
    image = image.cuda()
    cam = grad_cam(image)
    # image = image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    cv2_image = cv2.resize(cv2.imread(image_path), (256, 256))
    cv2_image = cv2.normalize(cv2_image, None, 0, 255, cv2.NORM_MINMAX)
    cam_image = [Drawer.overlay_cam_on_image(cv2_image, cam[0], use_rgb=True)]
    final_cam_image = cam_image
    cat_image = Drawer.concat(final_cam_image)
    cat_image.save(os.path.join(output_folder, file))