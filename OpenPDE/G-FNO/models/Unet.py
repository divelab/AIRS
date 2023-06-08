import torch.nn.functional as F
import torch
import escnn
import e2cnn
from utils import grid
# ----------------------------------------------------------------------------------------------------------------------
# Rotation Equivariant Unets
# ----------------------------------------------------------------------------------------------------------------------

class rot_conv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, N, act, activation=True, deconv=False,
                 last_deconv=False):
        super(rot_conv2d, self).__init__()
        r2_act = act

        feat_type_in = e2cnn.nn.FieldType(r2_act, input_channels * [r2_act.regular_repr])
        feat_type_hid = e2cnn.nn.FieldType(r2_act, output_channels * [r2_act.regular_repr])
        if not deconv:
            if activation:
                self.layer = e2cnn.nn.SequentialModule(
                    e2cnn.nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                                    padding=(kernel_size - 1) // 2),
                    e2cnn.nn.InnerBatchNorm(feat_type_hid),
                    e2cnn.nn.ReLU(feat_type_hid)
                )
            else:
                self.layer = e2cnn.nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                                             padding=(kernel_size - 1) // 2)
        else:
            if last_deconv:
                feat_type_in = e2cnn.nn.FieldType(r2_act, input_channels * [r2_act.regular_repr])
                feat_type_hid = e2cnn.nn.FieldType(r2_act, output_channels * [r2_act.trivial_repr])
                self.layer = e2cnn.nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride, padding=0)
            else:
                self.layer = e2cnn.nn.SequentialModule(
                    e2cnn.nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride, padding=0),
                    e2cnn.nn.InnerBatchNorm(feat_type_hid),
                    e2cnn.nn.ReLU(feat_type_hid)
                )

    def forward(self, x):
        return self.layer(x)


class rot_deconv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, N, last_deconv=False):
        super(rot_deconv2d, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N=N)
        self.conv2d = rot_conv2d(input_channels=input_channels, output_channels=output_channels, kernel_size=4,
                                 activation=True, stride=1, N=N, deconv=True, last_deconv=last_deconv, act=r2_act)
        self.feat_type = e2cnn.nn.FieldType(r2_act, input_channels * [r2_act.regular_repr])

    def pad(self, x):
        new_x = torch.zeros(x.shape[0], x.shape[1], x.shape[2] * 2 + 3, x.shape[3] * 2 + 3)
        new_x[:, :, :-3, :-3][:, :, ::2, ::2] = x
        new_x[:, :, :-3, :-3][:, :, 1::2, 1::2] = x
        new_x = e2cnn.nn.GeometricTensor(new_x, self.feat_type)
        return new_x

    def forward(self, x):
        out = self.pad(x).to(x.device)
        return self.conv2d(out)


class Unet_Rot(torch.nn.Module):
    def __init__(self, input_frames, output_frames, kernel_size, N):
        super(Unet_Rot, self).__init__()
        r2_act = e2cnn.gspaces.Rot2dOnR2(N=N)
        self.feat_type_in = e2cnn.nn.FieldType(r2_act, input_frames * [r2_act.trivial_repr])
        self.feat_type_in_hid = e2cnn.nn.FieldType(r2_act, 32 * [r2_act.regular_repr])
        self.feat_type_hid_out = e2cnn.nn.FieldType(r2_act, (16 + input_frames) * [r2_act.trivial_repr])
        self.feat_type_out = e2cnn.nn.FieldType(r2_act, output_frames * [r2_act.trivial_repr])

        self.conv1 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(self.feat_type_in, self.feat_type_in_hid, kernel_size=kernel_size, stride=2,
                      padding=(kernel_size - 1) // 2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_in_hid),
            e2cnn.nn.ReLU(self.feat_type_in_hid)
        )

        self.conv2 = rot_conv2d(32, 64, kernel_size=kernel_size, stride=1, N=N, act=r2_act)
        self.conv2_1 = rot_conv2d(64, 64, kernel_size=kernel_size, stride=1, N=N, act=r2_act)
        self.conv3 = rot_conv2d(64, 128, kernel_size=kernel_size, stride=2, N=N, act=r2_act)
        self.conv3_1 = rot_conv2d(128, 128, kernel_size=kernel_size, stride=1, N=N, act=r2_act)
        self.conv4 = rot_conv2d(128, 256, kernel_size=kernel_size, stride=2, N=N, act=r2_act)
        self.conv4_1 = rot_conv2d(256, 256, kernel_size=kernel_size, stride=1, N=N, act=r2_act)

        self.deconv3 = rot_deconv2d(256, 64, N)
        self.deconv2 = rot_deconv2d(192, 32, N)
        self.deconv1 = rot_deconv2d(96, 16, N, last_deconv=True)

        self.output_layer = e2cnn.nn.R2Conv(self.feat_type_hid_out, self.feat_type_out, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = x.permute(0, 3, 1, 2)
        x = e2cnn.nn.GeometricTensor(x, self.feat_type_in)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        out_deconv3 = self.deconv3(out_conv4.tensor)
        concat3 = torch.cat((out_conv3.tensor, out_deconv3.tensor), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2.tensor, out_deconv2.tensor), 1)
        out_deconv1 = self.deconv1(concat2)

        concat0 = torch.cat((x.tensor, out_deconv1.tensor), 1)
        concat0 = e2cnn.nn.GeometricTensor(concat0, self.feat_type_hid_out)
        out = self.output_layer(concat0)
        out = out.tensor
        out = out.permute(0, 2, 3, 1)
        return out.unsqueeze(-2)


class DownConvRot2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, act, pool='max'):
        super(DownConvRot2d, self).__init__()
        r2_act = act
        feat_type_in = e2cnn.nn.FieldType(r2_act, in_channels * [r2_act.regular_repr])
        if pool == 'max':
            self.pool = e2cnn.nn.PointwiseMaxPool(in_type=feat_type_in, kernel_size=2)
        elif pool == 'avg':
            self.pool = e2cnn.nn.PointwiseAvgPool(in_type=feat_type_in, kernel_size=2)
        else:
            raise ValueError
        self.conv = rot_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, N=N, act=act)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpConvRot2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, act, last_deconv=False):
        super(UpConvRot2d, self).__init__()
        r2_act = act
        feat_type_in = e2cnn.nn.FieldType(r2_act, in_channels * [r2_act.regular_repr])
        self.up = e2cnn.nn.R2Upsampling(in_type=feat_type_in, scale_factor=2, mode='bilinear')
        if not last_deconv:
            self.conv = rot_conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, N=N,act=act)
        else:
            feat_type_out = e2cnn.nn.FieldType(r2_act, out_channels * [r2_act.trivial_repr])
            self.conv = e2cnn.nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Unet_Rot_M(torch.nn.Module):
    def __init__(self, input_frames, output_frames, kernel_size, N, grid_type, pool='max', width=32):
        super(Unet_Rot_M, self).__init__()
        self.equiv = True
        r2_act = e2cnn.gspaces.Rot2dOnR2(N=N) # FlipRot2dOnR2(N=N)
        self.grid = grid(twoD=True, grid_type=grid_type)
        self.feat_type_in = e2cnn.nn.FieldType(r2_act, (input_frames + self.grid.grid_dim) * [r2_act.trivial_repr])
        self.feat_type_in_hid = e2cnn.nn.FieldType(r2_act, width * [r2_act.regular_repr])
        self.feat_type_hid_out = e2cnn.nn.FieldType(r2_act, (width // 2 + input_frames + self.grid.grid_dim) * [r2_act.trivial_repr])
        self.feat_type_out = e2cnn.nn.FieldType(r2_act, output_frames * [r2_act.trivial_repr])

        if pool == 'max':
            POOL = e2cnn.nn.PointwiseMaxPool
        elif pool == 'avg':
            POOL = e2cnn.nn.PointwiseAvgPool
        else:
            raise ValueError
        self.conv1 = e2cnn.nn.SequentialModule(
            POOL(in_type=self.feat_type_in, kernel_size=2),
            e2cnn.nn.R2Conv(self.feat_type_in, self.feat_type_in_hid, kernel_size=kernel_size, stride=1,
                            padding=(kernel_size - 1) // 2),
            e2cnn.nn.InnerBatchNorm(self.feat_type_in_hid),
            e2cnn.nn.ReLU(self.feat_type_in_hid)
        )

        self.conv2 = rot_conv2d(width, 2 * width, kernel_size=kernel_size, stride=1, N=N, act=r2_act)
        self.conv2_1 = rot_conv2d(2 * width, 2 * width, kernel_size=kernel_size, stride=1, N=N, act=r2_act)
        self.conv3 = DownConvRot2d(2 * width, 4 * width, kernel_size=kernel_size, N=N, pool=pool, act=r2_act)
        self.conv3_1 = rot_conv2d(4 * width, 4 * width, kernel_size=kernel_size, stride=1, N=N, act=r2_act)
        self.conv4 = DownConvRot2d(4 * width, 8 * width, kernel_size=kernel_size, N=N, pool=pool, act=r2_act)
        self.conv4_1 = rot_conv2d(8 * width, 8 * width, kernel_size=kernel_size, stride=1, N=N, act=r2_act)

        # Up sampling + conv
        self.deconv3 = UpConvRot2d(8 * width, 2 * width, kernel_size=kernel_size, N=N, act=r2_act)
        self.deconv2 = UpConvRot2d(6 * width, width, kernel_size=kernel_size, N=N, act=r2_act)
        self.deconv1 = UpConvRot2d(3 * width, width // 2, kernel_size=kernel_size, N=N, last_deconv=True, act=r2_act)

        self.output_layer = e2cnn.nn.R2Conv(self.feat_type_hid_out, self.feat_type_out, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = x.permute(0, 3, 1, 2)
        x = e2cnn.nn.GeometricTensor(x, self.feat_type_in)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_deconv3 = self.deconv3(out_conv4)
        concat3 = torch.cat((out_conv3.tensor, out_deconv3.tensor), 1)
        concat3 = e2cnn.nn.GeometricTensor(concat3, out_conv3.type + out_deconv3.type)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2.tensor, out_deconv2.tensor), 1)
        concat2 = e2cnn.nn.GeometricTensor(concat2, out_conv2.type + out_deconv2.type)
        out_deconv1 = self.deconv1(concat2)

        concat0 = torch.cat((x.tensor, out_deconv1.tensor), 1)
        concat0 = e2cnn.nn.GeometricTensor(concat0, self.feat_type_hid_out)
        out = self.output_layer(concat0)
        out = out.tensor
        out = out.permute(0, 2, 3, 1)
        return out.unsqueeze(-2)

################################################################
# 3d Unet
################################################################
class rot_conv3d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, N, act, activation=True):
        super(rot_conv3d, self).__init__()
        r3_act = act
        feat_type_in = escnn.nn.FieldType(r3_act, input_channels * [r3_act.regular_repr])
        feat_type_hid = escnn.nn.FieldType(r3_act, output_channels * [r3_act.regular_repr])
        if activation:
            self.layer = escnn.nn.SequentialModule(
                escnn.nn.R3Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                                padding=(kernel_size - 1) // 2),
                escnn.nn.IIDBatchNorm3d(feat_type_hid),
                escnn.nn.ReLU(feat_type_hid)
            )
        else:
            self.layer = escnn.nn.R3Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride,
                                         padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.layer(x)


class DownConvRot3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, act):
        super(DownConvRot3d, self).__init__()
        r3_act = act
        self.feat_type_in = escnn.nn.FieldType(r3_act, in_channels * [r3_act.regular_repr])
        self.conv = rot_conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, N=N, act=r3_act)

    def forward(self, x, pool_kernel_size=(2,2,2), pool_padding=(0,0,0)):
        x = F.max_pool3d(x.tensor, pool_kernel_size, padding=pool_padding)
        x = escnn.nn.GeometricTensor(x, self.feat_type_in)
        x = self.conv(x)
        return x


class UpConvRot3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, N, act, last_deconv=False):
        super(UpConvRot3d, self).__init__()
        r3_act = act
        self.feat_type_in = escnn.nn.FieldType(r3_act, in_channels * [r3_act.regular_repr])
        if not last_deconv:
            self.conv = rot_conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, N=N, act=r3_act)
        else:
            feat_type_out = escnn.nn.FieldType(r3_act, out_channels * [r3_act.trivial_repr])
            self.conv = escnn.nn.R3Conv(self.feat_type_in, feat_type_out, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x, scale_factor=None, out_size=None):
        x = F.interpolate(x.tensor, scale_factor=scale_factor, size=out_size, mode='trilinear')
        x = escnn.nn.GeometricTensor(x, self.feat_type_in)
        x = self.conv(x)
        return x

class Unet_Rot_3D(torch.nn.Module):
    def __init__(self, input_frames, output_frames, kernel_size, N, grid_type, width=32, debug=False):
        super(Unet_Rot_3D, self).__init__()
        self.equiv = True
        self.time = True
        self.debug = debug
        r3_act = escnn.gspaces.rot2dOnR3(n=N)
        self.grid = grid(twoD=False, grid_type=grid_type)
        self.feat_type_in = escnn.nn.FieldType(r3_act, (input_frames + self.grid.grid_dim) * [r3_act.trivial_repr])
        self.feat_type_in_hid = escnn.nn.FieldType(r3_act, width * [r3_act.regular_repr])
        self.feat_type_hid_out = escnn.nn.FieldType(r3_act, (width // 2 + input_frames + self.grid.grid_dim) * [r3_act.trivial_repr])
        self.feat_type_out = escnn.nn.FieldType(r3_act, output_frames * [r3_act.trivial_repr])

        self.conv1 = escnn.nn.SequentialModule(
            escnn.nn.R3Conv(self.feat_type_in, self.feat_type_in_hid, kernel_size=kernel_size, stride=1,
                            padding=(kernel_size - 1) // 2),
            escnn.nn.IIDBatchNorm3d(self.feat_type_in_hid),
            escnn.nn.ReLU(self.feat_type_in_hid)
        )

        self.conv2 = rot_conv3d(width, 2 * width, kernel_size=kernel_size, stride=1, N=N, act=r3_act)
        self.conv2_1 = rot_conv3d(2 * width, 2 * width, kernel_size=kernel_size, stride=1, N=N, act=r3_act)
        self.conv3 = DownConvRot3d(2 * width, 4 * width, kernel_size=kernel_size, N=N, act=r3_act)
        self.conv3_1 = rot_conv3d(4 * width, 4 * width, kernel_size=kernel_size, stride=1, N=N, act=r3_act)
        self.conv4 = DownConvRot3d(4 * width, 8 * width, kernel_size=kernel_size, N=N, act=r3_act)
        self.conv4_1 = rot_conv3d(8 * width, 8 * width, kernel_size=kernel_size, stride=1, N=N, act=r3_act)

        # Up sampling + conv
        self.deconv3 = UpConvRot3d(8 * width, 2 * width, kernel_size=kernel_size, N=N, act=r3_act)
        self.deconv2 = UpConvRot3d(6 * width, width, kernel_size=kernel_size, N=N, act=r3_act)
        self.deconv1 = UpConvRot3d(3 * width, width // 2, kernel_size=kernel_size, N=N, last_deconv=True, act=r3_act)

        self.output_layer = escnn.nn.R3Conv(self.feat_type_hid_out, self.feat_type_out, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def debug_print(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x):
        # collapse input times and channels; permute channels and time to the front; [B, S, S, T, C] -> [B, C, T, S, S]
        # "However, when working with voxel data, the (⋯,−Z,−Y,X) convention is used" - https://quva-lab.github.io/escnn/api/escnn.gspaces.html#group-actions-on-the-3d-space
        x = x.reshape((*x.shape[:-2], -1))
        x = self.grid(x)
        x = x.permute(0, 4, 3, 1, 2)
        self.debug_print('x', x.shape)
        x_pool = F.max_pool3d(x, kernel_size=(1, 2, 2))
        self.debug_print('x_pool', x_pool.shape)
        x_pool = escnn.nn.GeometricTensor(x_pool, self.feat_type_in)
        out_conv1 = self.conv1(x_pool)
        self.debug_print('out_conv1', out_conv1.shape)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        self.debug_print('out_conv2', out_conv2.shape)
        if x.shape[-3] == 10:
            pool_padding = (1, 0, 0)
        else:
            pool_padding = (0, 0, 0)
        out_conv3 = self.conv3_1(self.conv3(out_conv2, pool_kernel_size=2, pool_padding=pool_padding))
        self.debug_print('out_conv3', out_conv3.shape)
        out_conv4 = self.conv4_1(self.conv4(out_conv3, pool_kernel_size=2))
        self.debug_print('out_conv4', out_conv4.shape)

        # Up sampling
        out_deconv3 = self.deconv3(out_conv4, scale_factor=2)
        self.debug_print('out_deconv3', out_deconv3.shape)
        concat3 = torch.cat((out_conv3.tensor, out_deconv3.tensor), 1)
        concat3 = escnn.nn.GeometricTensor(concat3, out_conv3.type + out_deconv3.type)
        if x.shape[-3] in [9, 10]: # 9 for pdearena SWE and 10 for NS-sym
            out_deconv2 = self.deconv2(concat3, out_size=(x.shape[-3], out_deconv3.shape[-2] * 2, out_deconv3.shape[-1] * 2))
        else:
            out_deconv2 = self.deconv2(concat3, scale_factor=2)
        self.debug_print('out_deconv2', out_deconv2.shape)
        concat2 = torch.cat((out_conv2.tensor, out_deconv2.tensor), 1)
        concat2 = escnn.nn.GeometricTensor(concat2, out_conv2.type + out_deconv2.type)
        out_deconv1 = self.deconv1(concat2, scale_factor=(1, 2, 2))
        self.debug_print('out_conv1', out_deconv1.shape)
        concat0 = torch.cat((x, out_deconv1.tensor), 1)
        concat0 = escnn.nn.GeometricTensor(concat0, self.feat_type_hid_out)
        out = self.output_layer(concat0)
        out = out.tensor
        out = out.permute(0, 3, 4, 2, 1)  # [B, 1, T, S, S] -> [B, S, S, T, 1]
        self.debug_print('out', out.shape)
        return out


