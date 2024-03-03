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
import osr
np.set_printoptions(suppress=True)
from matplotlib.ticker import PercentFormatter
import socket
from torchviz import make_dot
from graphviz import Digraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_input_batch(isTraining = True, batch_size = 64, ismodel1=False):
    batch_size = batch_size
    if ismodel1:
        if isTraining:
            train_dataset = QuadhashDataset(training=True)
            dataloader_new = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            test_dataset = QuadhashDataset(training=False)
            dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        if isTraining:
            train_dataset = QuadhashDataset_model_2(training=True)
            dataloader_new = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            test_dataset = QuadhashDataset_model_2(training=False)
            dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return dataloader_new
#

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

class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, nf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)

        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                                             use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)


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


def train_model(generator, num_epochs = 200, batch_size = 64, dirno = 1):
    train_dl = get_input_batch(isTraining=True, batch_size=batch_size, ismodel1=False)
    test_dl = get_input_batch(isTraining=False, batch_size=32, ismodel1=False)

    # criterion = CustomLoss_science_guided_upweighing()
    criterion2 = CustomLoss()

    optimizer = torch.optim.Adam(generator.parameters(), lr=0.000001, betas=(0.5, 0.999))
    all_loss_train, all_loss_test = [],[]
    out_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(dirno) + "/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + 'samples', exist_ok=True)

    train_loss_file = out_path + 'train_loss.txt'
    test_loss_file = out_path + 'test_loss.txt'

    for epoch in range(1, num_epochs + 1):
        print("Training model on epoch: ", epoch,  "/", num_epochs)
        generator = generator.to(device).train()
        epoch_loss = 0.0

        for (input_img, target_img, _) in train_dl:
            input_img = input_img.to(torch.float32)
            target_img = target_img.to(torch.float32)

            input_img = input_img.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()
            outputs = generator(input_img)

            loss = criterion2(outputs, target_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dl)
        print(f"Epoch [{epoch}/{num_epochs}],Train Loss: {epoch_loss:.4f}")
        all_loss_train.append(epoch_loss)

        with open(train_loss_file, 'a') as f:
            f.write(f'{epoch_loss}\n')

        epoch_loss = 0.0
        generator = generator.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for (input_img, target_img, _) in test_dl:
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

        plt.plot(range(1, epoch + 1), all_loss_train, marker='o', color='black', markersize=2, linestyle='dotted', label='Training Loss')
        plt.plot(range(1, epoch + 1), all_loss_test, marker='x', color='red', markersize=2, linestyle='solid',
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
                input_sample, target_img, allquads = next(iter(test_dl))
                input_sample = input_sample.to(torch.float32)
                input_sample = input_sample.to(device)
                output_sample = generator(input_sample).cpu()
                print("out_sample_min", np.min(output_sample.numpy()))

                l1_error = generator_loss(output_sample[0], target_img[0])
                plot_targed_inferred(allquads[0], output_sample[0], target_img[0], l1_error, epoch, out_path, training=True)

                model_path = out_path + "model_weights.pth"
                x =  torch.randn(1, 17, 64, 64).to('cpu')
                traced_cell = torch.jit.trace(generator.to('cpu'), (x))
            torch.jit.save(traced_cell, model_path)

            # torch.save(generator, model_path)
            # with open(model_architecture_path, 'wb') as f:
            #     pickle.dump(generator, f)

    print("Training complete.")

def load_model_weights(folder):
    out_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/"
    model_path = out_path + "model_weights.pth"
    # model.load_state_dict(torch.load(model_path))
    # loaded_model = torch.load(model_path)
    loaded_model = torch.jit.load(model_path)
    return loaded_model

def save_as_np(output, quad, folder):
    out_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/tifs/"
    os.makedirs(out_path, exist_ok=True)
    out_path =  out_path + quad + ".npy"
    np.save(out_path, output)

def perform_inferences(folder, one_per_quad=False):
    model2 = load_model_weights(folder)
    model2 = model2.to(device).float()
    model2 = model2.eval()

    batch_size = 64
    out_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/"
    os.makedirs(out_path, exist_ok=True)
    test_dataset = QuadhashDataset_model_2(training=False, one_per_quad=one_per_quad)
    dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        for (input_sample, target_img, allquads) in dataloader_new:
            input_sample = input_sample.to(torch.float32).to(device)
            output_sample = model2(input_sample).cpu()
            # # for b in range(input_sample.shape[1]):
            # #     print("band: ", b, np.min(input_sample[:,b, :, :].numpy()))
            # output_sample[output_sample == -1] = np.nan
            # output_sample[output_sample<0] = 0
            # print("out", np.nanmin(output_sample)

            for i in range(output_sample.shape[0]):
            #     # save_as_np(output_sample[i], allquads[i], folder)
                l1_error = generator_loss(output_sample[0], target_img[0])
                plot_targed_inferred(allquads[i], output_sample[i], target_img[i], l1_error, i, out_path, training=False)
                # save_as_tif(output_sample_p, quad[i])

def plot_npy_files(folder=4):
    out_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(folder) + "/tifs/"
    for f in os.listdir(out_path):
        data = np.load(out_path + f)
        data[data==-1] = np.nan
        plt.imshow(data[0,:,:], vmin=0, vmax=1)
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

    inp_path2 = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/hru/split_14_okhla/"
    inp_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/3/"

    merge_paths = []
    date_not_found = []
    for quad in needed_quads:
        if os.path.exists(inp_path + quad + "/30m.tif"):
            merge_paths.append(inp_path + quad + "/30m.tif")
        else:
            if os.path.exists(inp_path2 + quad + "/20190716.tif"):
                print("mergin")
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
               './okhla_sm_model1.tif'] + \
              outs
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

    # plt.figure(figsize=(8, 6))
    # plt.hist(sorted_data, bins=10, density=True, alpha=0.7, color='green', edgecolor='black')
    # plt.xlabel('Value')
    # plt.ylabel('Percentage')
    # plt.title('Distribution of Data')
    # percentiles = [0, 2, 10, 30, 50, 70, 90, 98, 100]
    # for p in percentiles:
    #     percentile_value = np.percentile(sorted_data, p)
    #     plt.axvline(x=percentile_value, color='red', linestyle='dashed',
    #                 label=f'{p}th Percentile: {percentile_value:.2f}')
    # plt.legend(loc='best')
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    # plt.savefig('./values_dist_okhla_model_1_percentage.png')

    # inp_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/3/"
    # for quad in os.listdir(inp_path):
    #     im = gdal.Open(inp_path+ quad + "/30m.tif").ReadAsArray()
    #     print(np.min(im), np.max(im))

def plotting_model():

    # dot_graph = make_dot(output, params=dict(model1.named_parameters()))
    # dot_graph.attr("node", shape="record")
    # dot_graph.node("input", label="Input\n{0}".format(dummy_input.shape))
    # dot_graph.node("output", label="Output\n{0}".format(output.shape))
    #
    # dot_graph.render("./model_graph", format="png")

    dot = Digraph(comment='Model Architecture', format='png')

    # Register hooks to capture input/output shapes
    input_shapes = []
    output_shapes = []


    def hook(module, input, output):
        input_shape = str(list(input[0].shape))
        output_shape = str(list(output.shape))
        dot.node(str(id(module)), f'{module.__class__.__name__}\nInput: {input_shape}\nOutput: {output_shape}')


    def add_hooks(module):
        hook_handle = module.register_forward_hook(hook)
        hooks.append(hook_handle)


    hooks = []


    # Recursively add hooks to all submodules
    def recursive_add_hooks(module):
        for layer in module.children():
            if isinstance(layer, nn.Module):
                add_hooks(layer)
                recursive_add_hooks(layer)


    recursive_add_hooks(model)

    # Forward pass to capture input/output shapes
    with torch.no_grad():
        model(dummy_input)

    # Add edges between connected layers
    for module in model.children():
        for name, child in module.named_children():
            if len(list(child.children())) == 0:
                continue
            parent_id = id(module)
            child_id = id(child)
            dot.edge(str(parent_id), str(child_id))

    # Remove hooks
    for hook_handle in hooks:
        hook_handle.remove()

    # Save the graph visualization to a file
    dot.render('model_architecture_with_connections', format='png')

if __name__ == '__main__':
    model = UnetGenerator_model_2(input_nc=17, output_nc=1, nf=64, use_dropout=False).to(device).float()
    train_model(model, num_epochs=30, batch_size=64, dirno=3)
    # perform_inferences(folder=10, one_per_quad=False)
    # model2 = load_model_weights(4)
    # output = summary(model,  (17, 64, 64))
    # plot_npy_files()
