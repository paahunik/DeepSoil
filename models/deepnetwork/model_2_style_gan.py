import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import QuadhashDataset, QuadhashDataset_ONLY_INFER, QuadhashDataset_model_2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
import gdal
import subprocess
import csv
import osr
np.set_printoptions(suppress=True)
from matplotlib.ticker import PercentFormatter
import socket
from torchviz import make_dot
from graphviz import Digraph
import cv2

#Image-to-Image Translation with Conditional Adversarial Networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_input_batch(isTraining = True, batch_size = 64, ismodel1=False, cluster_no=11):
    batch_size = batch_size
    if ismodel1:
        if isTraining:
            train_dataset = QuadhashDataset(training=True, corrected_hru=False)
            dataloader_new = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            test_dataset = QuadhashDataset(training=False, corrected_hru=False)
            dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        if isTraining:
            train_dataset = QuadhashDataset_model_2(training=True, corrected_hru=True, cluster_no=cluster_no)
            dataloader_new = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            test_dataset = QuadhashDataset_model_2(training=False, corrected_hru=True, cluster_no=cluster_no)
            dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return dataloader_new

'''This unet architecture takes (64,64) and generates sm at 64,64,1'''
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        self.downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                  stride=2, padding=1, bias=False)
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.downnorm = norm_layer(inner_nc)
        self.uprelu = nn.ReLU(True)
        self.upnorm = norm_layer(outer_nc)
        self.last_activation = nn.ReLU(True)

        if outermost:
            self.start_conv = nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1)
            self.start_conv3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
            self.start_conv4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
            self.start_conv5 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

            self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1)
            self.down = [self.downconv]
            self.up = [self.uprelu, self.upconv, nn.ReLU(True)]
            self.model = nn.Sequential(*self.down, submodule, *self.up)
        elif innermost:
            self.upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=False)
            self.down = [self.downrelu, self.downconv]
            self.up = [self.uprelu, self.upconv, self.upnorm]
            self.model = nn.Sequential(*self.down, *self.up)
        else:
            self.upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=False)
            self.down = [self.downrelu, self.downconv, self.downnorm]
            self.up = [self.uprelu, self.upconv, self.upnorm]

            if use_dropout:
                self.model = nn.Sequential(*self.down, submodule, *self.up, nn.Dropout(0.5))
            else:
                self.model = nn.Sequential(*self.down, submodule, *self.up)

    def forward(self, x):
        if self.outermost:
            inp2 = x[:, -4:,:,:]
            input_conv = self.start_conv(inp2)
            input_conv = self.uprelu(input_conv)
            input_conv = self.start_conv3(input_conv)
            input_conv = self.uprelu(input_conv)
            input_conv = self.start_conv4(input_conv)
            input_conv = self.uprelu(input_conv)

            my_model = self.model(x)
            concatenated = torch.cat([input_conv, my_model], 1)
            newc = self.start_conv5(concatenated)
            newc = self.last_activation(newc)
            return newc
        else:
            return torch.cat([x, self.model(x)], 1)


'''This unet architecture takes (64,64) and generates sm 64,64,1'''
class UnetGenerator_model_2(nn.Module):

    def __init__(self, input_nc, output_nc, nf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator_model_2, self).__init__()

        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, innermost=True,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)

def generator_loss(generated_image, target_img):
    l1_loss = nn.L1Loss()
    mask = (target_img != -1).float()
    l1_l = l1_loss(generated_image * mask, target_img * mask)
    return l1_l

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        mask = (target != -1).float()
        loss = self.criterion(input * mask, target * mask)
        return loss

class CustomLoss_science_guided_upweighing(nn.Module):
    def __init__(self, mae_weight=0.5, fss_weight=0.2, lc_weight=0.6, prc_weight=0.2, lambda_val=0.5):
        super(CustomLoss_science_guided_upweighing, self).__init__()
        self.mae_criterion = nn.L1Loss()
        self.mae_weight = mae_weight
        self.prc_weight = prc_weight * lambda_val
        self.fss_weight = fss_weight * lambda_val
        self.land_cover_penalty_weight = lc_weight * lambda_val

    # https://crops.extension.iastate.edu/blog/mark-licht-mike-castellano-sotirios-archontoulis/facts-soil-moisture-benchmarking-tool#:~:text=A%20value%20of%200%20means,drought%20and%20excess%20moisture%20stress.
    def calculate_fss(self, input, target, mask=None, window_size=32, threshold_dry_opt=0.2, threshold_opt_ex=0.4):
        if mask is None:
           mask = (target != -1).float()

        threshold_do = threshold_dry_opt * (window_size ** 2) # dry optimum
        threshold_oe = threshold_opt_ex * (window_size ** 2) # optimum excess

        h, w = input.shape[2], input.shape[3]
        fss_scores = []
        for i in range(h - window_size + 1):
            for j in range(w - window_size + 1):
                window_input = input[:, :,  i:i+window_size, j:j+window_size]
                window_target = target[:, :,  i:i+window_size, j:j+window_size]
                window_mask = mask[:, :,  i:i+window_size, j:j+window_size]

                input_sum = (window_input * window_mask).sum(dim=(2, 3))
                target_sum = (window_target * window_mask).sum(dim=(2, 3))

                input_wet = (input_sum >= threshold_oe).float()
                target_wet = (target_sum >= threshold_oe).float()

                input_opt = ((input_sum >= threshold_do) & (input_sum < threshold_oe)).float()
                target_opt = ((target_sum >= threshold_do) & (input_sum < threshold_oe)).float()

                input_dry = (input_sum < threshold_do).float()
                target_dry = (target_sum < threshold_do).float()

                fss_components = (input_wet * target_wet) + (input_opt * target_opt) + (input_dry * target_dry)
                fss_score = fss_components.sum(dim=1) / (window_size ** 2)
                fss_scores.append(1 - fss_score)

        fss_scores_tensor = torch.stack(fss_scores)
        fss_scores_tensor = fss_scores_tensor.mean(dim=0)  # where a higher score indicates better agreement.
        fss_scores_tensor = fss_scores_tensor.mean(dim=0)
        return fss_scores_tensor

    def calculate_prc_loss(self, input, target, mask, precip_band):
        input = input * mask
        target = target * mask
        precip_band = precip_band * mask
        return torch.mean(torch.abs(input - target) * precip_band)

    def calculate_landcover_loss(self, input, target, mask, land_cover):
        if land_cover== None:
            return torch.tensor([0])

        unique_landc = torch.unique(land_cover)
        input = input * mask
        target = target * mask
        stds = []
        for val in unique_landc:
            mask_l = (land_cover == val).float()
            new_i = input * mask_l
            new_t = target * mask_l
            std_dev = new_i.std()
            std_dev_t = new_t.std()
            stds.append(std_dev_t - std_dev)

        std_tensor = torch.tensor(stds)
        return std_tensor.mean()

    def forward(self, input, target, land_cover=None, precip_band = None, only_mae_nlcd=False):
        mask = (target != -1).float()
        if self.land_cover_penalty_weight > 0:
            landc_loss = self.calculate_landcover_loss(input, target, mask, land_cover)
        else:
            landc_loss = 0

        mae_loss = self.mae_criterion(input * mask, target * mask)
        if only_mae_nlcd:
            return 0.6 * mae_loss + 0.4 * landc_loss, mae_loss
        else:
            if self.fss_weight > 0:
                fss_loss = self.calculate_fss(input, target, mask)
            else:
                fss_loss = 0
            if self.prc_weight > 0:
                prc_loss = self.calculate_prc_loss(input, target, mask, precip_band=precip_band)
            else:
                prc_loss = 0
            return self.mae_weight * mae_loss + self.fss_weight * fss_loss + self.prc_weight * prc_loss+\
                   self.land_cover_penalty_weight * landc_loss, mae_loss

def plot_targed_inferred(allquads, output_sample, target_img, l1_error, epoch, out_path, training=True):

    plt.subplot(1, 2, 1)
    plt.title("Target Image: \n" + allquads)

    output_sample = np.ma.masked_equal(output_sample.permute(1, 2, 0).numpy(), -1)
    target_img = np.ma.masked_equal(target_img.permute(1, 2, 0).numpy(), -1)
    plt.imshow(target_img)
    plt.axis('off')

    colorbar_target = plt.colorbar()
    # colorbar_target.set_label('Target Color Bar Label')

    plt.subplot(1, 2, 2)
    plt.title("Generated Image \n L1  loss: " + str(round(l1_error.item(), 4)))
    plt.imshow(output_sample)
    plt.axis('off')

    colorbar_generated = plt.colorbar()
    # colorbar_generated.set_label('Generated Color Bar Label')
    if training:
        plt.savefig(out_path + 'samples/' + str(epoch) + '.png')
    else:
        os.makedirs(out_path + 'testing', exist_ok=True)
        plt.savefig(out_path + 'testing/' + str(epoch) + '.png')
    plt.close()

def load_losses_train_test(dirno):
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(dirno) + "/"

    train_loss_file = out_path + 'train_loss.txt'
    test_loss_file = out_path + 'test_loss.txt'

    with open(train_loss_file, 'r') as f:
        train_loss_lines = f.readlines()
    train_loss_values = [float(line.strip()) for line in train_loss_lines]

    with open(test_loss_file, 'r') as f:
        test_loss_lines = f.readlines()
    test_loss_values = [float(line.strip()) for line in test_loss_lines]

    return train_loss_values, test_loss_values

def plot_inputs():
    test_dl = get_input_batch(isTraining=False, batch_size=16, ismodel1=False)
    for (input_img, target_img, quad) in train_dl:
        plt.subplot(1, 2, 1)
        plt.title("Target Image: \n" + quad)

        output_sample = np.ma.masked_equal(output_sample.permute(1, 2, 0).numpy(), -1)
        target_img = np.ma.masked_equal(target_img.permute(1, 2, 0).numpy(), -1)
        plt.imshow(target_img)
        plt.axis('off')

        colorbar_target = plt.colorbar()
        # colorbar_target.set_label('Target Color Bar Label')

        plt.subplot(1, 2, 2)
        plt.title("Generated Image \n L1  loss: " + str(round(l1_error.item(), 4)))
        plt.imshow(output_sample)
        plt.axis('off')

def train_model(generator, num_epochs = 200, batch_size = 64, dirno = 1,lr= 0.0001, cluster_no=11):
    train_dl = get_input_batch(isTraining=True, batch_size=batch_size, ismodel1=False, cluster_no=cluster_no)
    test_dl = get_input_batch(isTraining=False, batch_size=32, ismodel1=False, cluster_no=cluster_no)

    criterion1 = CustomLoss_science_guided_upweighing(mae_weight=1.0, lambda_val=0)
    criterion2 = CustomLoss()

    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(dirno) + "/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + 'samples', exist_ok=True)
    all_loss_train, all_loss_test = [], []
    train_loss_file = out_path + 'train_loss.txt'
    test_loss_file = out_path + 'test_loss.txt'

    if os.path.exists(train_loss_file):
        with open(train_loss_file, "r") as file:
            start_ep = sum(1 for line in file)
        with open(train_loss_file, "r") as file:
            for line in file:
                all_loss_train.append(float(line.rstrip()))
    else:
            start_ep = 0

    if os.path.exists(test_loss_file):
        with open(test_loss_file, "r") as file:
            all_loss_test.append(float(line) for line in file)
        with open(test_loss_file, "r") as file:
            for line in file:
                all_loss_test.append(float(line.rstrip()))

    for epoch in range(start_ep, num_epochs + 1):
        print("Training model on epoch: ", epoch,  "/", num_epochs)
        generator = generator.to(device).train()
        epoch_loss = 0.0

        for (input_img, target_img, _, _) in train_dl:
            input_img = input_img.to(torch.float32)
            target_img = target_img.to(torch.float32)

            input_img = input_img.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()
            outputs = generator(input_img)
            # if epoch < 150:
            loss, _ = criterion1(outputs, target_img, input_img[:,-4, :, :], input_img[:,3,:,:], only_mae_nlcd=False)
            # else:
            #     loss, _ = criterion1(outputs, target_img, input_img[:, -4, :, :], input_img[:, 3, :, :], only_mae_nlcd=True)
            # loss, _ = criterion1(outputs, target_img, input_img[:, -4, :, :], input_img[:, 3, :, :], only_mae_nlcd=True)
            # loss = criterion2(outputs, target_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)
        print(f"Epoch [{epoch}/{num_epochs}],Train Loss: {epoch_loss:.4f}")
        all_loss_train.append(epoch_loss)

        with open(train_loss_file, 'a') as f:
            f.write(f'{epoch_loss}\n')

        epoch_loss = 0.0
        generator = generator.eval()
        with torch.no_grad():
            for (input_img, target_img, _, _) in test_dl:
                input_img = input_img.to(torch.float32)
                target_img = target_img.to(torch.float32)
                input_img = input_img.to(device)
                target_img = target_img.to(device)
                outputs = generator(input_img)
                loss = criterion2(outputs, target_img)
                epoch_loss += loss.item()

        epoch_loss /= len(test_dl)
        print(f"Epoch [{epoch}/{num_epochs}], Test Loss: {epoch_loss:.4f}")
        all_loss_test.append(epoch_loss)

        plt.plot(range(0, epoch + 1), all_loss_train, marker='o', color='#984ea3', markersize=2, linestyle='dotted', label='Training Loss')
        plt.plot(range(0, epoch + 1), all_loss_test, marker='x', color='#999999', markersize=2, linestyle='solid',
                 label='Testing Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(out_path + 'training_test_loss.png')
        plt.close()

        with open(test_loss_file, 'a') as f:
            f.write(f'{epoch_loss}\n')

        if epoch % 5 == 0:
            print("Saving sample test data and loss")
            generator = generator.eval()
            with torch.no_grad():
                input_sample, target_img, allquads, _ = next(iter(test_dl))
                input_sample = input_sample.to(torch.float32)
                input_sample = input_sample.to(device)
                output_sample = generator(input_sample).cpu()

                l1_error = generator_loss(output_sample[0], target_img[0])
                plot_targed_inferred(allquads[0], output_sample[0], target_img[0], l1_error, epoch, out_path, training=True)

                model_path = out_path + "model_weights.pth"
                x = torch.randn(1, 17, 64, 64).to('cpu')
                traced_cell = torch.jit.trace(generator.to('cpu'), (x))
            torch.jit.save(traced_cell, model_path)

    print("Training complete.")

def load_model_weights(folder):
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/"
    model_path = out_path + "model_weights.pth"
    # model.load_state_dict(torch.load(model_path))
    # loaded_model = torch.load(model_path)
    loaded_model = torch.jit.load(model_path)
    return loaded_model

def save_as_np(output, quad, folder, month=None):
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/tifs/"
    os.makedirs(out_path, exist_ok=True)
    out_path = out_path + quad + "_" + month + ".npy"
    np.save(out_path, output)

def load_station(q, d):
    station_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/station_data/split/5/"
    new_p = station_path + q + "/" + d.split(".")[0] + ".txt"
    lat_station, lon_station, sm_true = [], [], []
    if os.path.exists(new_p):
        with open(new_p, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                lat_station.append(float(row[0]))
                lon_station.append(float(row[1]))
                sm_true.append(float(row[2]))
        return lat_station, lon_station, sm_true
    else:
        return None, None, None


def get_closest_station_window(x_station, y_station, sm_true, sm_hru, arr_sm_hru):
    error_list = []
    window_size = 16
    half_window = window_size // 2
    station_vals, out_vals = [], []

    for t_s in range(len(x_station)):
        x_pixel = int((x_station[t_s] - sm_hru.GetGeoTransform()[0]) / sm_hru.GetGeoTransform()[1])-1
        y_pixel = int((y_station[t_s] - sm_hru.GetGeoTransform()[3]) / sm_hru.GetGeoTransform()[5])-1

        x_start = max(0, x_pixel - half_window)
        x_end = min(arr_sm_hru.shape[1], x_pixel + half_window + 1)
        y_start = max(0, y_pixel - half_window)
        y_end = min(arr_sm_hru.shape[0], y_pixel + half_window + 1)

        arr_sm_hru[arr_sm_hru == -1] = np.nan
        # print("Station: ",  sm_true[t_s], arr_sm_hru[y_pixel, x_pixel])
        try:
            window_data = arr_sm_hru[y_start:y_end, x_start:x_end]
            # window_data = arr_sm_hru
            val = np.min(np.abs(window_data - sm_true[t_s]))

            station_vals.append(sm_true[t_s])
            # val = np.min(np.abs(window_data - sm_true[t_s]))
            # val = np.nanmean(window_data)
            if np.isnan(val):
                val = np.nanmean(arr_sm_hru)
                err = sm_true[t_s] - val
                print("Getting all image average")
            else:
                err = val

            min_error_index = np.argmin(np.abs(window_data - sm_true[t_s]))
            out_vals.append(window_data[min_error_index])

        except IndexError:
           return None, None, None

        error_list.append(abs(err))
    return error_list, station_vals, out_vals

def perform_inferences_stations(folder, training=False, one_per_quad=False, iscorrected=False):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()
    if iscorrected:
        inp_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_corrected_14/"
    else:
        inp_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_14/"

    output_geotiff_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/tiffs/"
    os.makedirs(output_geotiff_path, exist_ok=True)

    batch_size = 32
    test_dataset = QuadhashDataset_model_2(training=training, one_per_quad=one_per_quad, corrected_hru=iscorrected, cluster_no=11)
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    all_err, all_stations, all_out_vals = [], [], []

    with torch.no_grad():
        for (input_sample, target_img, allquads, dates) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu().numpy()

            for i in range(output_sample.shape[0]):
                d = dates[i]
                q = allquads[i]
                lat_s, lon_s, sm_s = load_station(q, d)
                if sm_s is not None:
                    if os.path.exists(inp_path + q + "/" + d):
                        sm_hru = gdal.Open(inp_path + q + "/" + d)

                        if sm_hru is None:
                            continue
                        geotransform = sm_hru.GetGeoTransform()
                        projection = sm_hru.GetProjection()
                        resized_out_image = cv2.resize(output_sample[i][0], (sm_hru.RasterXSize, sm_hru.RasterYSize))
                        driver = gdal.GetDriverByName('GTiff')
                        b_dataset = driver.Create(output_geotiff_path + q + "_" + d, sm_hru.RasterXSize, sm_hru.RasterYSize, 1,
                                                  gdal.GDT_Float32)  # Change data type as needed

                        b_dataset.SetProjection(projection)
                        b_dataset.SetGeoTransform(geotransform)
                        band = b_dataset.GetRasterBand(1)
                        band.WriteArray(resized_out_image)
                        b_dataset = None
                        resized_out_image[resized_out_image == -1] = np.nan
                        errs, station_vals, out_vals = get_closest_station_window(lon_s, lat_s, sm_s, sm_hru, resized_out_image)

                        all_err.extend(errs)
                        all_stations.extend(station_vals)
                        all_out_vals.extend(out_vals)
                        sm_hru = None

                    else:
                        continue


    print("len: ", len(all_stations), len(all_out_vals))
    print("Average error SM: ", np.mean(np.array(all_err)), " average std: ", np.std(np.array(all_err)), " on sample count: ", len(all_err))


def perform_inferences_station_smap_hru(folder, training=False, one_per_quad=False, iscorrected=False):
    output_geotiff_path =  "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_14/"
    # output_geotiff_path =  "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/smap_9km/split_14/"

    batch_size = 32
    test_dataset = QuadhashDataset_model_2(training=training, one_per_quad=one_per_quad, corrected_hru=iscorrected, cluster_no=11)
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    all_err = []
    with torch.no_grad():
        for (input_sample, target_img, allquads, dates) in dataloader_new:
            for i in range(input_sample.shape[0]):
                d = dates[i]
                q = allquads[i]
                lat_s, lon_s, sm_s = load_station(q, d)
                # q = q[:12]
                if sm_s is not None:
                    if os.path.exists(output_geotiff_path + q + "/" + d):
                        sm_hru = gdal.Open(output_geotiff_path + q + "/" + d)

                        if sm_hru is None:
                            continue

                        resized_out_image = sm_hru.ReadAsArray()
                        resized_out_image[resized_out_image == -9999] = np.nan

                        print("\n", q, d)
                        errs = get_closest_station_window(lon_s, lat_s, sm_s, sm_hru, resized_out_image)
                        all_err.extend(errs)
                        sm_hru = None
                    else:
                        continue
    print("Average error SM: ", np.mean(np.array(all_err)), " average std: ", np.std(np.array(all_err)), " on sample count: ", len(all_err))

def perform_inferences(folder, one_per_quad=False):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()

    batch_size = 32
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/"
    os.makedirs(out_path, exist_ok=True)
    test_dataset = QuadhashDataset_model_2(cluster_no=11, training=False, one_per_quad=one_per_quad, corrected_hru=False)
    # test_dataset = QuadhashDataset(training=True, corrected_hru=True)
    total_loss = []
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    count = 0
    with torch.no_grad():
        for (input_sample, target_img, allquads, months) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu()
            # # for b in range(input_sample.shape[1]):
            # #     print("band: ", b, np.min(input_sample[:,b, :, :].numpy()))
            # output_sample[output_sample == -1] = np.nan
            # output_sample[output_sample<0] = 0
            # print("out", np.nanmin(output_sample)

            for i in range(output_sample.shape[0]):
                count +=1
            #     # save_as_np(output_sample[i], allquads[i], folder)
                l1_error = generator_loss(output_sample[i], target_img[i])
                total_loss.append(round(l1_error.item(), 4))
                # plot_targed_inferred(allquads[i], output_sample[i], target_img[i], l1_error, count, out_path, training=False)
                # save_as_np(output_sample[i], allquads[i], folder, months[i])
    print("Average loss on test data: ", np.mean(np.array(total_loss)))

def plot_npy_files(folder=1):
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/tifs/"
    for f in os.listdir(out_path):
        data = np.load(out_path + f)
        data[data==-1] = np.nan
        plt.imshow(data[0,:,:])
        plt.savefig("./plts/" + f + ".png")
        plt.close()

def remove_empty_folders():
    path_dir = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/3/"
    tot = len(os.listdir(path_dir))
    count = 0
    for q in os.listdir(path_dir):
        if len(os.listdir(path_dir + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(path_dir + q)
    print(count, "/", tot)

def merge_okhla():
    file_path = './okhla_quad_needed.txt'
    with open(file_path, 'r') as file:
        needed_quads = [line.strip() for line in file]

    inp_path2 = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_14_okhla/"
    inp_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/3/"

    merge_paths = []
    date_not_found = []

    for quad in needed_quads:
        if os.path.exists(inp_path + quad + "/30m.tif"):
            merge_paths.append(inp_path + quad + "/30m.tif")
        else:
            if os.path.exists(inp_path2 + quad + "/20190716.tif"):
                print("merging")
                merge_paths.append(inp_path2 + quad + "/20190716.tif")

    print("Merging :", len(merge_paths))

    batch_size = 10000
    total_batches = (len(merge_paths) + batch_size - 1) // batch_size
    outs = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = merge_paths[start_idx:end_idx]

        command = ['gdal_merge.py', '-n', str(-1), '-a_nodata', '-1', '-o', './okhla_sm' + "_batch_"+ str(batch_idx)  +".tif"] + \
                  batch
        outs.append('./okhla_sm' + "_batch_" + str(batch_idx)  + ".tif")
        subprocess.call(command)

    command = ['gdal_merge.py', '-n', str(-1), '-a_nodata', '-1', '-o',
               './okhla_sm_model1.tif'] + outs
    subprocess.call(command)

def merge_plot_state_sm_maps():
    merge_okhla()
    im = gdal.Open("./okhla_sm_model1.tif").ReadAsArray()
    im[im < 0] = np.nan
    # im[im > 0.8] = np.nan
    filtered_data = im[~np.isnan(im)]
    sorted_data = np.sort(filtered_data)
    print(np.min(sorted_data))

    percentile_2 = np.percentile(filtered_data, 2)
    percentile_90 = np.percentile(filtered_data, 99)

    print("Value at 2%:", percentile_2)
    print("Value at 99%:", percentile_90)

    plt.figure(figsize=(8, 6))
    plt.hist(sorted_data, bins=40, density=True, alpha=0.9, color='lightgrey', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Data')
    percentiles = [2, 10, 30, 50, 70, 90, 99]
    colormap = plt.cm.turbo

    for p in percentiles:
        percentile_value = np.percentile(sorted_data, p)
        color = colormap(p / 100)
        plt.axvline(x=percentile_value, color=color, linestyle='dashed',
                    label=f'{p}th Percentile: {percentile_value:.2f}')
    plt.legend(loc='best')
    plt.savefig('./values_dist_okhla_model_1.png')



if __name__ == '__main__':
    model = UnetGenerator_model_2(input_nc=17, output_nc=1, nf=64, use_dropout=False).to(device).float()
    # model = load_model_weights(2)
    model = model.to(device).float()
    # perform_inferences_stations(7, iscorrected=True, one_per_quad=False, training=False)
    # perform_inferences_station_smap_hru(11, iscorrected=False, one_per_quad=True, training=False)
    train_model(model, num_epochs=1501, batch_size=32, dirno=25, lr= 0.0001, cluster_no=11)
    # out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(12) + "/"


    # dir 12: window 8, 0.2 fss, 0.4 land, 0.6: mae


    # perform_inferences(folder=10, one_per_quad=True)
    # model2 = load_model_weights(4)
    # output = summary(model,  (17, 64, 64))
    # plot_npy_files(1)






