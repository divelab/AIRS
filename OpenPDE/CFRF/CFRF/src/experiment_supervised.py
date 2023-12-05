import random
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from models import ModeFormer, SRCNN, RDN, Wrapper
from models_FNO import FNO2d
from RCAN import RCAN
from DataLoader import PDEDataset, train_valid_test_split
from torch.utils.data import DataLoader
from utils import save_img, pdeLoss, downsample, pdeLossSWE, pdeLossDiffReact
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn as nn
import shutil
import h5py
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment(object):
    def __init__(self, args):
        self.args = args

        params = {
            "num_layers": args.layers,
            "input_dim": 1,
            "hidden_dim": args.emb_dim,
            "output_dim": 1,
            "residual": args.residual,
            "scale_factor": args.scale_factor,
            "upsample": args.upsample,
        }

        if args.model == 'SRCNN':
            print('Use SRCNN!!!')
            if self.args.field_dim > 1:
                initial_step = args.sw * self.args.field_dim
            else:
                initial_step =  args.sw
            params_SRCNN = {
            "num_layers": args.layers,
            "input_dim": 1,
            "hidden_dim": args.emb_dim,
            "output_dim": 1,
            "residual": args.residual and (args.PDE != 'iCFD'),
            "scale_factor": args.scale_factor,
            "upsample": args.upsample,
            }
            if self.args.seperate_prediction:
                self.model = Wrapper(params_SRCNN, field_dim=self.args.field_dim, model_name=args.model).to(device)
            else:
                self.model = SRCNN(**params_SRCNN).to(device)
        else:
            raise ValueError(f"Model {args.model} not recognized")


        self.num_params = sum(p.numel() * (1 + p.is_complex()) for p in self.model.parameters())
        print(f"\n# Params: {self.num_params}")


        # Initialize loss, Adam optimizer and scheduler
        self.reg_criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_init, weight_decay=2e-4)
        self.scheduler = StepLR(self.optimizer, step_size=300, gamma=0.25)


    def run(self):

        # load data
        datapath = os.path.join(self.args.data_path, self.args.dataname)
        x, y = self.load_data(datapath, scale_factor=self.args.scale_factor)
        x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(x, y)

        if self.args.PDE == 'iCFD':
            assert(x_train.shape[1] == 500)
            if self.args.subset:
                # self.data = data[0:1, 0:(data.shape[1] // 4), :, :]
                if self.args.data_frac == "1/16":
                    x_train_valid = x_train[0:1, :, :, :]
                    y_train_valid = y_train[0:1, :, :, :]
                elif self.args.data_frac == "1/32":
                    x_train_valid = x_train[0:1, 0:(x_train.shape[1] // 2), :, :]
                    y_train_valid = y_train[0:1, 0:(y_train.shape[1] // 2), :, :]
                elif self.args.data_frac == "1/64":
                    x_train_valid = x_train[0:1, 0:(x_train.shape[1] // 4), :, :]
                    y_train_valid = y_train[0:1, 0:(y_train.shape[1] // 4), :, :]
                elif self.args.data_frac == "1/8":
                    x_train_valid = x_train[0:2, :, :, :]
                    y_train_valid = y_train[0:2, :, :, :]
                elif self.args.data_frac == "1/128":
                    x_train_valid = x_train[0:1, 0:(x_train.shape[1] // 8), :, :]
                    y_train_valid = y_train[0:1, 0:(y_train.shape[1] // 8), :, :]  
                else:
                    raise ValueError(f"data_frac is not right!")
            elif self.args.unsupervised:
                x_train_valid = x_train
                y_train_valid = y_train
            else:
                raise ValueError("Training config not recognized")

            time_steps = x_train_valid.shape[1]
            x_train = x_train_valid[:,0:int(time_steps * 0.8),:,:]
            y_train = y_train_valid[:,0:int(time_steps * 0.8),:,:]

            x_valid = x_train_valid[:,int(time_steps * 0.8):,:,:]
            y_valid = y_train_valid[:,int(time_steps * 0.8):,:,:]


            self.train_set = PDEDataset(x_train, labels=y_train, training=True, sw=self.args.sw, subset=self.args.subset, data_frac=self.args.data_frac)
            self.valid_set = PDEDataset(x_valid, labels=y_valid, training=False, sw=self.args.sw)
            self.test_set = PDEDataset(x_test, labels=y_test, training=False, sw=self.args.sw)
        elif self.args.PDE in ['swe', 'diff-react']:
            assert(x_train.shape[1] == 101)
            assert(x_train.shape[0] == 80)
            if self.args.subset:
                # self.data = data[0:1, 0:(data.shape[1] // 4), :, :]
                if self.args.data_frac == "1/80":
                    x_train_valid = x_train[0:1, ...]
                    y_train_valid = y_train[0:1, ...]
                elif self.args.data_frac == "1/20":
                    x_train_valid = x_train[0:4, ...]
                    y_train_valid = y_train[0:4, ...]
                else:
                    raise ValueError(f"data_frac is not right!")
            else:
                raise ValueError("Training config not recognized")

            if self.args.data_frac == "1/80":
                time_steps = x_train_valid.shape[1]
                x_train = x_train_valid[:, 0:int(time_steps * 0.8), ...]
                y_train = y_train_valid[:, 0:int(time_steps * 0.8), ...]

                x_valid = x_train_valid[:, int(time_steps * 0.8):, ...]
                y_valid = y_train_valid[:, int(time_steps * 0.8):, ...]
            elif self.args.data_frac == "1/20":

                num_traj = x_train_valid.shape[0]
                x_train = x_train_valid[0:num_traj-1, :, ...]
                y_train = y_train_valid[0:num_traj-1, :, ...]

                x_valid = x_train_valid[num_traj-1:, :, ...]
                y_valid = y_train_valid[num_traj-1:, :, ...]
            else:
                raise ValueError(f"data_frac is not right!")    


            self.train_set = PDEDataset(x_train, labels=y_train, training=True, sw=self.args.sw, subset=self.args.subset, data_frac=self.args.data_frac)
            self.valid_set = PDEDataset(x_valid, labels=y_valid, training=False, sw=self.args.sw)
            self.test_set = PDEDataset(x_test, labels=y_test, training=False, sw=self.args.sw)
        else:
            raise ValueError("Data not recognized")


        # initialize data loader
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        valid_loader = DataLoader(self.valid_set, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        if self.args.cosine_scheduler:
            num_training_steps = self.args.epochs * len(train_loader)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_training_steps)


        if self.args.mode == 'test':

            print("Start Test mode!!!")
            checkpoint_dir = os.path.join(self.args.working_dir, self.args.checkpoint_dir)
            
            print('Loading checkpoint!!!')
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint_bst_mae.pt'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.eval(test_loader, test=True)
            return


        if self.args.log_dir != '' and self.args.mode == 'train':
            if self.args.debug:
                log_dir = os.path.join(self.args.log_dir, self.args.suffix + '_debug')
            else:
                log_dir = os.path.join(self.args.log_dir, self.args.cur_time + self.args.suffix)
            # log_dir = os.path.join(args.log_dir, args.suffix)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            else:
                # shutil.rmtree(log_dir)
                for f in os.listdir(log_dir):
                    os.remove(os.path.join(log_dir, f))
            
            writer = SummaryWriter(log_dir=log_dir)


        best_valid_mse = 1000
        best_valid_mse_epoch = 0
        best_valid_pde = 1000
        best_valid_pde_epoch = 0

        for epoch in range(1, self.args.epochs + 1):
            start = time.time()
            print("=====Epoch {}".format(epoch))
            print('Training...')
            train_loss, train_pred = self.train(train_loader)

            print('Evaluating...')
            valid_mse, valid_HR_pde_loss, eval_data = self.eval(valid_loader)
            # valid_mae = 0

            print("Epoch {:d}, Train_loss: {:.5f}, Validation_mae: {:.5f}, elapse: {:.5f}".
                    format(epoch, train_loss['total_loss'], valid_mse, time.time() - start))

            if self.args.log_dir != '':
                writer.add_scalar('valid/L2_loss', valid_mse, epoch)
                writer.add_scalar('valid/HR_pde_loss', valid_HR_pde_loss, epoch)
                writer.add_scalar('train/total_loss', train_loss['total_loss'], epoch)
                writer.add_scalar('train/HR_pde_loss', train_loss['HR_pde_loss'], epoch)
                writer.add_scalar('train/HR2HR_mse', train_loss['HR2HR_mse'], epoch)

                valid_pred_fig, (ax1, ax2) = plt.subplots(1,2)

                ax1.set_title("HR reconstructed")
                ax2.set_title("HR ground truth")
                # ax3.set_title("HR residual")

                if eval_data['pred'].dim() == 5:
                    ax1.imshow(eval_data['pred'][0, -1, :, :, 0], cmap='twilight')
                    ax2.imshow(eval_data['label'][0, -1, :, :, 0], cmap='twilight')
                else:
                    ax1.imshow(eval_data['pred'][0, -1, :, :], cmap='twilight')
                    ax2.imshow(eval_data['label'][0, -1, :, :], cmap='twilight')
                # ax3.imshow(eval_data['HR_residual'][0][1], cmap='twilight')
 
                writer.add_figure('valid/pred_fig', valid_pred_fig, epoch)


                train_pred_fig, (ax1, ax2) = plt.subplots(1,2)

                ax1.set_title("HR reconstructed")
                ax2.set_title("HR ground truth")
                if train_pred['pred'].dim() == 5:
                    ax1.imshow(train_pred['pred'][0, 1, :, :, 0], cmap='twilight')
                    ax2.imshow(train_pred['label'][0, 1, :, :, 0], cmap='twilight')
                else:
                    ax1.imshow(train_pred['pred'][0][1], cmap='twilight')
                    ax2.imshow(train_pred['label'][0][1], cmap='twilight')

                writer.add_figure('train/pred_fig', train_pred_fig, epoch)

            if self.args.choose_best_model == 'pde':
                if valid_HR_pde_loss < best_valid_pde:
                    best_valid_pde = valid_HR_pde_loss
                    best_valid_pde_epoch = epoch

                    if self.args.checkpoint_dir != '':
                        print('Saving checkpoint...')
                        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                      'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'num_params': self.num_params,
                                      'mse': valid_mse, 'pde': valid_HR_pde_loss}
                        checkpoint_dir = os.path.join(self.args.working_dir, self.args.checkpoint_dir)
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir)
                        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_mae.pt'))

                print(f'Best validation PDE so far: {best_valid_pde}, epoch: {best_valid_pde_epoch}')

            elif self.args.choose_best_model == 'mse':
                if valid_mse < best_valid_mse:
                    best_valid_mse = valid_mse
                    best_valid_mse_epoch = epoch

                    if self.args.checkpoint_dir != '':
                        print('Saving checkpoint...')
                        checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                      'optimizer_state_dict': self.optimizer.state_dict(),
                                      'scheduler_state_dict': self.scheduler.state_dict(),
                                      'num_params': self.num_params,
                                      'mse': valid_mse}
                        checkpoint_dir = os.path.join(self.args.working_dir, self.args.checkpoint_dir)
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir)
                        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_mae.pt'))

                print(f'Best validation MSE so far: {best_valid_mse}, epoch: {best_valid_mse_epoch}')

            else:
                raise ValueError(f"choose_best_model is not right!")

            # if epoch % 20 == 0:
            #     print('====== Start testing !!! =====')
            #     self.eval(test_loader, test=True)


        # test
        print('====== Start testing !!! =====')
        checkpoint_dir = os.path.join(self.args.working_dir, self.args.checkpoint_dir)
        print('Loading best checkpoint!!!')
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint_bst_mae.pt'))
        print('Best checkpoint is from: ', checkpoint['epoch'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.eval(test_loader, test=True)

        return

    def train(self, loader):
        self.model.train()
        total_loss = 0
        total_HR_pde_loss = 0

        total_mse = 0

        pred_list = []
        ground_truth_list = []

        t = tqdm(loader, desc="Iteration")
        for step, (inputs, labels) in enumerate(t):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            if self.args.PDE == 'iCFD':
                low_s = inputs.shape[-1]
                high_s = labels.shape[-1]
                scale = int(high_s // low_s)
            elif self.args.PDE in ['swe', 'diff-react']:
                low_s = inputs.shape[-2]
                high_s = labels.shape[-2]
                scale = int(high_s // low_s)
            else:
                raise ValueError(f"equation is not right!")

            
            
            if self.args.PDE == 'iCFD':
                HR_reconstructed = self.model(inputs)
            elif self.args.PDE in ['swe', 'diff-react']:
                B, T, s_x, s_y, D = inputs.shape
                inputs = inputs.permute(0,1,4,2,3).reshape(B, T*D, s_x, s_y)
                HR_reconstructed = self.model(inputs)

                HR_reconstructed = HR_reconstructed.reshape(B, T, D, high_s, high_s).permute(0,1,3,4,2)
            else:
                raise ValueError(f"equation is not right!")

            
            # calculate loss
            # LR_pde_loss = pdeLoss(LR_corrected, re=1000.0, dt=0.02, s=low_s, L=self.args.L)
            if self.args.PDE == 'iCFD':
                HR_pde_loss = pdeLoss(HR_reconstructed, re=1000.0, dt=0.02, s=high_s, L=self.args.L)
            elif self.args.PDE == 'swe':
                g = 1
                dt = 0.01
                dx = 0.0390625
                dy = 0.0390625
                HR_pde_loss = pdeLossSWE(data=HR_reconstructed, dx=dx, dy=dy, dt=dt, g=g)
            elif self.args.PDE == 'diff-react':
                dx = 0.015625
                dy = 0.015625
                dt = 0.05
                HR_pde_loss = pdeLossDiffReact(data=HR_reconstructed, dx=dx, dy=dy, dt=dt)
            else:
                raise ValueError(f"equation is not right!")
            
            mse = self.reg_criterion(HR_reconstructed, labels)
            
            
            if self.args.supervised_loss == 'mse':
                loss = mse
            elif self.args.supervised_loss == 'mse+pde':
                loss = HR_pde_loss + mse
            else:
                raise ValueError(f"loss {self.args.supervised_loss} not recognized")


            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            t.set_description(f"PDEloss {HR_pde_loss:.3f}, mse {mse:.3f}")

            pred_list.append(HR_reconstructed.detach().cpu())
            ground_truth_list.append(labels.detach().cpu())

            total_loss += loss.detach().cpu()
            total_HR_pde_loss += HR_pde_loss.detach().cpu()
            total_mse += mse.detach().cpu()
            if self.args.cosine_scheduler:
                self.scheduler.step()
            # if self.scheduler.optimizer.param_groups[0]['lr'] > 5e-4:
            #     self.scheduler.step()

        pred = torch.cat(pred_list, dim=0)
        label = torch.cat(ground_truth_list, dim=0)

        train_pred = {'pred': pred, 'label':label}
        
        loss_dict = {
            'total_loss': total_loss / (step + 1),
            'HR_pde_loss': total_HR_pde_loss / (step + 1),
            'HR2HR_mse': total_mse / (step + 1),
        }

        return loss_dict, train_pred

    def eval(self, loader, test=False):
        self.model.eval()

        pred_list = []
        ground_truth_list = []
        interp_nearest_list = []
        interp_bicubic_list = []


        if test:
            num_seqs = self.test_set.num_seqs
        else:
            num_seqs = self.valid_set.num_seqs


        for step, (inputs, labels) in enumerate(tqdm(loader, desc="Iteration")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if self.args.PDE == 'iCFD':
                low_s = inputs.shape[-1]
                high_s = labels.shape[-1]
                scale = int(high_s // low_s)
            elif self.args.PDE in ['swe', 'diff-react']:
                low_s = inputs.shape[-2]
                high_s = labels.shape[-2]
                scale = int(high_s // low_s)
            else:
                raise ValueError(f"equation is not right!")

            with torch.no_grad():
                # LR_corrected, HR_reconstructed, LR_residual, HR_residual, _ = self.model(inputs)
                if self.args.PDE == 'iCFD':
                    HR_reconstructed = self.model(inputs)
                elif self.args.PDE in ['swe', 'diff-react']:
                    B, T, s_x, s_y, D = inputs.shape
                    inputs = inputs.permute(0,1,4,2,3).reshape(B, T*D, s_x, s_y)
                    HR_reconstructed = self.model(inputs)
                    HR_reconstructed = HR_reconstructed.reshape(B, T, D, high_s, high_s).permute(0,1,3,4,2)
                else:
                    raise ValueError(f"equation is not right!")

            # pred = HR_reconstructed * std + mean
            pred = HR_reconstructed

            pred_list.append(pred)
            ground_truth_list.append(labels)


            # baseline method: two interpolation, nearest and bicubic
            if self.args.PDE == 'iCFD':
                nearest_upsampler = nn.Upsample(scale_factor=scale, mode='nearest')
                bicubic_upsampler = nn.Upsample(scale_factor=scale, mode='bicubic')
                output_nearest = nearest_upsampler(inputs)
                output_bicubic = bicubic_upsampler(inputs)
                interp_nearest_list.append(output_nearest)
                interp_bicubic_list.append(output_bicubic)
            elif self.args.PDE in ['swe', 'diff-react']:
                nearest_upsampler = nn.Upsample(scale_factor=scale, mode='nearest')
                bicubic_upsampler = nn.Upsample(scale_factor=scale, mode='bicubic')
                # inputs = inputs.permute(0,1,4,2,3).reshape(B, T*D, s_x, s_y) # already reshaped above
                output_nearest = nearest_upsampler(inputs).reshape(B, T, D, high_s, high_s).permute(0,1,3,4,2)
                output_bicubic = bicubic_upsampler(inputs).reshape(B, T, D, high_s, high_s).permute(0,1,3,4,2)
                
                interp_nearest_list.append(output_nearest)
                interp_bicubic_list.append(output_bicubic)                
            else:
                raise ValueError(f"equation is not right!")                

        if self.args.PDE == 'iCFD': 
            pred = torch.cat(pred_list, dim=0).cpu().view(num_seqs, -1, 256, 256)
            label = torch.cat(ground_truth_list, dim=0).cpu().view(num_seqs, -1, 256, 256)

            interp_nearest = torch.cat(interp_nearest_list, dim=0).cpu().view(num_seqs, -1, 256, 256)
            interp_bicubic = torch.cat(interp_bicubic_list, dim=0).cpu().view(num_seqs, -1, 256, 256)

        elif self.args.PDE in ['swe', 'diff-react']:
            pred = torch.cat(pred_list, dim=0).cpu().reshape(num_seqs, -1, self.args.HR_resolution, self.args.HR_resolution, self.args.field_dim)
            label = torch.cat(ground_truth_list, dim=0).cpu().reshape(num_seqs, -1, self.args.HR_resolution, self.args.HR_resolution, self.args.field_dim)

            interp_nearest = torch.cat(interp_nearest_list, dim=0).cpu().reshape(num_seqs, -1, self.args.HR_resolution, self.args.HR_resolution, self.args.field_dim)
            interp_bicubic = torch.cat(interp_bicubic_list, dim=0).cpu().reshape(num_seqs, -1, self.args.HR_resolution, self.args.HR_resolution, self.args.field_dim)            
        else:
            raise ValueError(f"equation is not right!")      

        eval_data = {'pred': pred, 
                     'label':label, 
                     'nearest': interp_nearest,
                     'bicubic': interp_bicubic,
                     }

        
        # if test:
        save_img(eval_data, self.args, test=test)

        
        # calculate loss and accuracy
        # mse = torch.mean((torch.cat(pred_list, dim=0) - torch.cat(ground_truth_list, dim=0))**2)
        mse = self.reg_criterion(pred, label)
        nearest_mse = self.reg_criterion(interp_nearest, label)
        bicubic_mse = self.reg_criterion(interp_bicubic, label)

        # calculate PDE loss
        if self.args.PDE == 'iCFD': 
            HR_pde_loss = pdeLoss(pred, re=1000.0, dt=0.02, s=256, L=self.args.L)
            nearest_pde_loss = pdeLoss(interp_nearest, re=1000.0, dt=0.02, s=256, L=self.args.L)
            bicubic_pde_loss = pdeLoss(interp_bicubic, re=1000.0, dt=0.02, s=256, L=self.args.L)
        elif self.args.PDE == 'swe':
            g = 1
            dt = 0.01
            dx = 0.0390625
            dy = 0.0390625
            HR_pde_loss = pdeLossSWE(data=pred, dx=dx, dy=dy, dt=dt, g=g)
            nearest_pde_loss = pdeLossSWE(data=interp_nearest, dx=dx, dy=dy, dt=dt, g=g)
            bicubic_pde_loss = pdeLossSWE(data=interp_bicubic, dx=dx, dy=dy, dt=dt, g=g)
        elif self.args.PDE == 'diff-react':
            dx = 0.015625
            dy = 0.015625
            dt = 0.05
            HR_pde_loss = pdeLossDiffReact(data=pred, dx=dx, dy=dy, dt=dt)
            nearest_pde_loss = pdeLossDiffReact(data=interp_nearest, dx=dx, dy=dy, dt=dt)
            bicubic_pde_loss = pdeLossDiffReact(data=interp_bicubic, dx=dx, dy=dy, dt=dt)
        else:
            raise ValueError(f"equation is not right!")
        
        print('===== SRNet =====')
        print(f'L2 loss: {mse}')
        print(f'PDE loss: {HR_pde_loss}')
        print('===== Nearest =====')
        print(f'L2 loss: {nearest_mse}')
        print(f'PDE loss: {nearest_pde_loss}')
        print('===== Bicubic =====')
        print(f'L2 loss: {bicubic_mse}')
        print(f'PDE loss: {bicubic_pde_loss}')

        # if test:
        #     with open(os.path.join(self.args.working_dir, 'test_result.txt'), 'w') as f:
        #         f.write('===== SRNet =====\n')
        #         f.write(f'L2 loss: {mse}\n')
        #         f.write(f'PDE loss: {HR_pde_loss}\n')
        #         f.write('===== Nearest =====\n')
        #         f.write(f'L2 loss: {nearest_mse}\n')
        #         f.write(f'PDE loss: {nearest_pde_loss}\n')
        #         f.write('===== Bicubic =====\n')
        #         f.write(f'L2 loss: {bicubic_mse}\n')
        #         f.write(f'PDE loss: {bicubic_pde_loss}\n')

        return mse, HR_pde_loss, eval_data
    

    def load_data(self, datapath=None, scale_factor=None):
        def load_iCFD(fname=None):

            mat_contents = torch.load(fname)
            data = mat_contents['u'].permute(0,3,1,2) # seq x frames x H x W

            return data

        def load_swe(data_path):

            h5_file = h5py.File(data_path, "r")
            data_list = []
            seeds = np.arange(0,100)
            for seed in seeds:
                seed = str(seed).zfill(4)
                data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 3]
                data = torch.from_numpy(data).unsqueeze(0)
                data_list.append(data)
            
            data = torch.cat(data_list, dim=0)

            return data

        def load_diff_react(data_path):
            if data_path[-2:] == 'pt':
                data = torch.load(data_path)
            elif data_path[-2:] == 'h5':
                h5_file = h5py.File(data_path, "r")
                data_list = []
                seeds = np.arange(0,100)
                for seed in seeds:
                    seed = str(seed).zfill(4)
                    data = np.array(h5_file[f"{seed}/data"], dtype="f")  # dim = [101, 128, 128, 2]
                    data = torch.from_numpy(data).unsqueeze(0)
                    data_list.append(data)
                
                data = torch.cat(data_list, dim=0)

            return data


        print('Loading data!!!')
        
        if self.args.PDE == 'iCFD':
            
            HR_path = f'../data/ns_V0.001_L2pi_N20_T10_HR_256.pt'
            LR_path = f'../data/ns_V0.001_L2pi_N20_T10_LR_64.pt'
            
            LR_data = load_iCFD(LR_path)
            HR_data = load_iCFD(HR_path)
        elif self.args.PDE == 'swe':

            HR_path = f'../data/2D_rdb_NA_NA_HUV_HR_{self.args.HR_resolution}_T1_N100.h5'
            LR_path = f'../data/2D_rdb_NA_NA_HUV_LR_{self.args.LR_resolution}_T1_N100.h5'       

            LR_data = load_swe(LR_path)
            HR_data = load_swe(HR_path)

        elif self.args.PDE == 'diff-react':

            HR_path = '../data/2D_diff-react_NA_NA_HR_128_T5_N100.h5'
            LR_path = '../data/2D_diff-react_NA_NA_LR_16_T5_N100.pt'
            
            HR_data = load_diff_react(HR_path)
            LR_data = load_diff_react(LR_path)
            

            assert LR_data.shape[2] == self.args.LR_resolution
            assert LR_data.shape[3] == self.args.LR_resolution

        else:
            raise ValueError(f"equation is not right!")
        
        print('Finished loading data!!!')

        return LR_data, HR_data