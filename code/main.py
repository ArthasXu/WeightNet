# python = 3.8.20   pytorch = 1.11.0 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import OpenEXR
import Imath
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--trainDir", type=str, default='C:\\Users\\Admin\\Desktop\\weightnet\\data\\input')
parser.add_argument("--trainGtDir", type=str, default='C:\\Users\\Admin\\Desktop\\weightnet\\data\\target')
parser.add_argument("--testDir", type=str, default='C:\\Users\\Admin\\Desktop\\weightnet\\test\\input')
parser.add_argument("--testGtDir", type=str, default='C:\\Users\\Admin\\Desktop\\weightnet\\test\\target')
parser.add_argument("--testOutDir", type=str, default='C:\\Users\\Admin\\Desktop\\weightnet\\results\\test_out')
parser.add_argument("--checkPointDir", type=str, default='C:\\Users\\Admin\\Desktop\\weightnet\\results\\pretrained_ckpt\\weight_net_model.pth')
parser.add_argument("--logDir", type=str, default='"C:\\Users\\Admin\\Desktop\\weightnet\\logs\\fit"')
# parser.add_argument("--input_feature_dim", type=int, default=INPUT_FEATS_DIM) #
# parser.add_argument("--epochs", type=int, default=50)
# parser.add_argument("--batchSize", type=int, default=4)
# parser.add_argument("--patchSize", type=int, default=128)
# parser.add_argument("--stride", type=int, default=64)
# parser.add_argument("--lr", type=float, default=1e-4)
# parser.add_argument("--kernelSize", type=int, default=KERNEL_SIZE)
# parser.add_argument("--olsKernelSize", type=int, default=51)
# parser.add_argument("--kernelSizeSqr", type=int, default=KERNEL_SIZE_SQR)
# parser.add_argument("--retrain", action="store_true")
# parser.add_argument("--loadEpoch", type=int, default=50)
#*********** TRAIN / TEST / VALIDATION *******************#
parser.add_argument("--mode", "-m", type=str, required=True) # train or test

# parser.add_argument("--valid", action="store_true")

args, unknown = parser.parse_known_args()


class WeightNet(nn.Module):
    def __init__(self):
        super(WeightNet, self).__init__()
        
        # Example of some convolution layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        
        # Decoder layers for upsampling
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)

        # Final layer to get a weight map (1 channel output)
        self.output_layer = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        inputs = inputs.permute(0, 1, 4, 3, 2)
        inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[3], inputs.shape[4])
        # inputs is of shape [batch_size, 24, 384, 384]
        x = torch.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))

        # Decoder
        x = self.upsample1(x)
        x = torch.relu(self.conv4(x))
        x = self.upsample2(x)
        x = torch.relu(self.conv5(x))

        # Output: a single channel weight map (alpha)
        alpha = torch.sigmoid(self.output_layer(x))  # [batch_size, 1, 384, 384]

        return alpha


class EXRDataset(Dataset):
    def __init__(self, trainDir, trainGtDir, scene_names, spp_values):
        self.trainDir = trainDir
        self.trainGtDir = trainGtDir
        self.scene_names = scene_names
        self.spp_values = spp_values

    def __len__(self):
        return len(self.scene_names) * len(self.spp_values)

    def __getitem__(self, idx):
        scene_idx = idx // len(self.spp_values)
        spp_idx = idx % len(self.spp_values)

        scene_name = self.scene_names[scene_idx]
        spp_value = self.spp_values[spp_idx]

        pt_path = os.path.join(self.trainDir, f"{scene_name}_pt_{spp_value}.exr")
        sppm_path = os.path.join(self.trainDir, f"{scene_name}_sppm_{spp_value}.exr")
        reference_path = os.path.join(self.trainGtDir, f"{scene_name}_bdpt_512.exr")

        # pt_path = "G:\\BaiduSyncdisk\\research\\neural_rendering\\network\\jscombiner\\data\\test\\input\\curly-hair\\curly-hair_16.exr"
        # sppm_path = "G:\\BaiduSyncdisk\\research\\neural_rendering\\network\\jscombiner\\data\\test\\input\\curly-hair\\biased\\curly-hair_16_afgsa.exr"
        # reference_path = "G:\\BaiduSyncdisk\\research\\neural_rendering\\network\\jscombiner\\data\\test\\target\\curly-hair.exr"

        print('pt_path: ', pt_path)
        print('sppm_path: ', sppm_path)
        print('reference_path: ', reference_path)

        pt_data = self.read_exr(pt_path)
        sppm_data = self.read_exr(sppm_path)
        reference = self.read_exr(reference_path)

        # print('pt_data: ', pt_data)
        # print('sppm_data: ', sppm_data)
        # print('reference: ', reference)

        variance_pt = pt_data['Variance']     
        albedo = pt_data['Albedo']         
        normal = pt_data['N']           

        variance_sppm = sppm_data['Variance'] 
        bias_sppm = sppm_data['Bias']    
        bias_pt = torch.zeros_like(torch.tensor(bias_sppm))

        img_unbiased = torch.tensor(pt_data['color']).unsqueeze(0).float()  
        img_biased = torch.tensor(sppm_data['color']).unsqueeze(0).float() 

        # print("img_unbiased shape:", img_unbiased.shape)
        # print("img_biased shape:", img_biased.shape)
        # print("variance_pt shape:", torch.tensor(variance_pt).unsqueeze(0).shape)
        # print("albedo shape:", torch.tensor(albedo).unsqueeze(0).shape)
        # print("normal shape:", torch.tensor(normal).unsqueeze(0).shape)
        # print("variance_sppm shape:", torch.tensor(variance_sppm).unsqueeze(0).shape)
        # print("bias_pt shape:", bias_pt.shape)
        # print("bias_sppm shape:", torch.tensor(bias_sppm).unsqueeze(0).shape)

        inputs = torch.cat([img_unbiased, 
                            img_biased, 
                            torch.tensor(variance_pt).unsqueeze(0), 
                            torch.tensor(albedo).unsqueeze(0), 
                            torch.tensor(normal).unsqueeze(0),
                            torch.tensor(variance_sppm).unsqueeze(0), 
                            torch.tensor(bias_pt).unsqueeze(0),
                            torch.tensor(bias_sppm).unsqueeze(0)], dim=0)
        
        reference_tensor = torch.tensor(reference['color']).unsqueeze(0).float()

        return inputs, reference_tensor
    
    def read_exr(self, file_path):
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        print("Reading file: ", file_path)

        """Read EXR file and return its image data as a numpy array."""
        try:
            exr_file = OpenEXR.InputFile(file_path)
            print(f"EXR file opened successfully: {file_path}")
        except Exception as e:
            print(f"Failed to open EXR file: {e}")
            return None
            
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        channel_names = exr_file.header()['channels'].keys()

        img_dict = {}

        for channel_name in channel_names:
            float_type = Imath.PixelType(Imath.PixelType.FLOAT)
            channel_data = exr_file.channel(channel_name, float_type)
            img_dict[channel_name] = np.frombuffer(channel_data, dtype=np.float32).reshape((height, width))

        if 'R' in img_dict and 'G' in img_dict and 'B' in img_dict:
            r = img_dict['R']
            g = img_dict['G']
            b = img_dict['B']
            rgb = np.stack([r, g, b], axis=-1)   
            img_dict['color'] = rgb

        if 'A1-L.R' in img_dict and 'A1-L.G' in img_dict and 'A1-L.B' in img_dict:
            _r = img_dict['A1-L.R']
            _g = img_dict['A1-L.G']
            _b = img_dict['A1-L.B']
            _rgb = np.stack([_r, _g, _b], axis=-1)   
            img_dict['color'] = _rgb

        if 'Albedo.R' in img_dict and 'Albedo.G' in img_dict and 'Albedo.B' in img_dict:
            albedo_r = img_dict['Albedo.R']
            albedo_g = img_dict['Albedo.G']
            albedo_b = img_dict['Albedo.B']
            albedo_rgb = np.stack([albedo_r, albedo_g, albedo_b], axis=-1)   
            img_dict['Albedo'] = albedo_rgb

        if 'N.X' in img_dict and 'N.Y' in img_dict and 'N.Z' in img_dict:
            n_r = img_dict['N.X']
            n_g = img_dict['N.Y']
            n_b = img_dict['N.Z']
            n_rgb = np.stack([n_r, n_g, n_b], axis=-1)   
            img_dict['N'] = n_rgb

        if 'Variance.R' in img_dict and 'Variance.G' in img_dict and 'Variance.B' in img_dict:
            variance_r = img_dict['Variance.R']
            variance_g = img_dict['Variance.G']
            variance_b = img_dict['Variance.B']
            variance_rgb = np.stack([variance_r, variance_g, variance_b], axis=-1)   
            img_dict['Variance'] = variance_rgb

        if 'A2-Bias.R' in img_dict and 'A2-Bias.G' in img_dict and 'A2-Bias.B' in img_dict:
            bias_r = img_dict['A2-Bias.R']
            bias_g = img_dict['A2-Bias.G']
            bias_b = img_dict['A2-Bias.B']
            bias_rgb = np.stack([bias_r, bias_g, bias_b], axis=-1)   
            img_dict['Bias'] = bias_rgb

        if 'A3-Variance.R' in img_dict and 'A3-Variance.G' in img_dict and 'A3-Variance.B' in img_dict:
            var_sppm__r = img_dict['A3-Variance.R']
            var_sppm__g = img_dict['A3-Variance.G']
            var_sppm__b = img_dict['A3-Variance.B']
            var_sppm__rgb = np.stack([var_sppm__r, var_sppm__g, var_sppm__b], axis=-1)   
            img_dict['Variance'] = var_sppm__rgb

        return img_dict


def train_model(train_loader, model, device, epochs=50, batch_size=16, writer=None):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, reference) in enumerate(train_loader):
            # Move data to device (e.g., GPU)
            inputs, reference = inputs.to(device), reference.to(device)

            optimizer.zero_grad()

            # Forward pass through the model to get the alpha (weight map)
            alpha = model(inputs)  # Output shape: [batch_size, 1, 384, 384]

            # Get the unbiased and biased images and reference
            img_unbiased = inputs[:, 0, :, :, :].permute(0, 3, 1, 2)
            img_biased = inputs[:, 1, :, :, :].permute(0, 3, 1, 2)
            reference = reference[:, 0, :, :, :].permute(0, 3, 1, 2)

            # Merge the images based on alpha (weight map)
            I_combined = alpha * img_unbiased + (1 - alpha) * img_biased  # Shape: [batch_size, 3, 384, 384]

            # Calculate the loss between the merged image and the reference
            loss = criterion(I_combined, reference)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")

            running_loss += loss.item()
            # 写入TensorBoard
            if writer:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        
        if writer:
            writer.add_scalar('Loss/epoch', running_loss / len(train_loader), epoch)

def save_exr(filename, data):
    """
    将图像数据保存为 EXR 格式。
    参数:
        filename: 保存的文件名
        data: 图像数据，形状为 (C, H, W)，通道为 1 或 3
    """
    height, width = data.shape[1], data.shape[2]
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    
    if data.shape[0] == 1:
        header['channels'] = {'Y': half_chan}
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({'Y': data[0].astype(np.float32).tobytes()})
    elif data.shape[0] == 3:
        header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({
            'R': data[0].astype(np.float32).tobytes(),
            'G': data[1].astype(np.float32).tobytes(),
            'B': data[2].astype(np.float32).tobytes()
        })
    exr.close()

def generate_and_save_results(model_path, test_loader, output_dir, device):
    model = WeightNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad(): 
        for i, (inputs, reference) in enumerate(test_loader):
            inputs = inputs.to(device)

            alpha = model(inputs) 

            img_unbiased = inputs[:, 0, :, :, :].permute(0, 3, 1, 2).to(device)  # [batch_size, 3, 384, 384]
            img_biased = inputs[:, 1, :, :, :].permute(0, 3, 1, 2).to(device)


            I_combined = alpha * img_unbiased + (1 - alpha) * img_biased  # [batch_size, 3, 384, 384]

            for j in range(inputs.size(0)):
                alpha_np = alpha[j].cpu().numpy()
                combined_np = I_combined[j].cpu().numpy()

                save_exr(os.path.join(output_dir, f'weight_map_{i * inputs.size(0) + j}.exr'), alpha_np)
                save_exr(os.path.join(output_dir, f'combined_image_{i * inputs.size(0) + j}.exr'), combined_np)
                print(f'Saved EXR weight map and combined image {i * inputs.size(0) + j}.')


def main():
    scene_names = ['bmw', 'book', 'zeroday-frame120']
    spp_values = [4, 16, 64]
    
    trainDir = args.trainDir
    trainGtDir = args.trainGtDir
    testOutDir = args.testOutDir
    checkPointDir = args.checkPointDir
    log_dir = args.logDir
    mode = args.mode
    
    if mode == 'train':
        dataset = EXRDataset(trainDir, trainGtDir, scene_names, spp_values)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TensorBoard
        writer = SummaryWriter(log_dir)

        model = WeightNet().to(device)  # Initialize model
        train_model(train_loader, model, device, epochs=50, batch_size=16, writer=writer)

        torch.save(model.state_dict(), checkPointDir)

        # Close TensorBoard
        writer.close()
    elif mode == 'test':
        test_dataset = EXRDataset(trainDir, trainGtDir, scene_names, spp_values)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generate_and_save_results(checkPointDir, test_loader, testOutDir, device)

if __name__ == '__main__':
    main()