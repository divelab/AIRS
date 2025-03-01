import torch
import numpy as np
import os
import glob
import random
import matplotlib
import imageio

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.molecule_builder import get_bond_order


##############
### Files ####
###########-->


def save_xyz_file(path, one_hot, positions, atom_decoder, id_from=0,
                  name='molecule', batch_mask=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if batch_mask is None:
        batch_mask = torch.zeros(len(one_hot))

    for batch_i in torch.unique(batch_mask):
        cur_batch_mask = (batch_mask == batch_i)
        n_atoms = int(torch.sum(cur_batch_mask).item())
        f = open(path + name + '_' + "%03d.xyz" % (batch_i + id_from), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[cur_batch_mask], dim=1)
        batch_pos = positions[cur_batch_mask]
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = atom_decoder[atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, batch_pos[atom_i, 0], batch_pos[atom_i, 1], batch_pos[atom_i, 2]))
        f.close()


def load_molecule_xyz(file, dataset_info):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(dataset_info['atom_decoder']))
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.xyz")
    if shuffle:
        random.shuffle(files)
    return files


# <----########
### Files ####
##############
def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    # for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color,
                    linewidth=0,
                    alpha=alpha)
    # # calculate vectors for "vertical" circle
    # a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    # b = np.array([0, 1, 0])
    # b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (
    #             1 - np.cos(rot))
    # ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle='dashed')
    # horiz_front = np.linspace(0, np.pi, 100)
    # ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')
    # vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    # ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u),
    #         a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle='dashed')
    # ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front),
    #         b[1] * np.cos(vert_front),
    #         a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front), color='k')
    #
    # ax.view_init(elev=elev, azim=0)


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color,
                  dataset_info):
    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    # ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2
    # areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)  # , linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                           dataset_info['atom_decoder'][atom_type[j]]
            s = (atom_type[i], atom_type[j])

            draw_edge_int = get_bond_order(dataset_info['atom_decoder'][s[0]],
                                           dataset_info['atom_decoder'][s[1]],
                                           dist)
            line_width = 2

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    # linewidth_factor = draw_edge_int  # Prop to number of
                    # edges.
                    linewidth_factor = 1
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        linewidth=line_width * linewidth_factor,
                        c=hex_bg_color, alpha=alpha)


def plot_data3d(positions, atom_type, dataset_info, camera_elev=0,
                camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                  hex_bg_color, dataset_info)

    # if 'qm9' in dataset_info['name']:
    max_value = positions.abs().max().item()

    # axis_lim = 3.2
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    # elif dataset_info['name'] == 'geom':
    #     max_value = positions.abs().max().item()
    #
    #     # axis_lim = 3.2
    #     axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    #     ax.set_xlim(-axis_lim, axis_lim)
    #     ax.set_ylim(-axis_lim, axis_lim)
    #     ax.set_zlim(-axis_lim, axis_lim)
    # elif dataset_info['name'] == 'pdbbind':
    #     max_value = positions.abs().max().item()
    #
    #     # axis_lim = 3.2
    #     axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    #     ax.set_xlim(-axis_lim, axis_lim)
    #     ax.set_ylim(-axis_lim, axis_lim)
    #     ax.set_zlim(-axis_lim, axis_lim)
    # else:
    #     raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_data3d_uncertainty(
        all_positions, all_atom_types, dataset_info, camera_elev=0,
        camera_azim=0,
        save_path=None, spheres_3d=False, bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    for i in range(len(all_positions)):
        positions = all_positions[i]
        atom_type = all_atom_types[i]
        plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                      hex_bg_color, dataset_info)

    if 'qm9' in dataset_info['name']:
        max_value = all_positions[0].abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = all_positions[0].abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 2 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'pdbbind':
        max_value = all_positions[0].abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 2 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.

        ax.imshow(im)

    plt.show()


def visualize(path, dataset_info, max_num=25, wandb=None, spheres_3d=False):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        positions, one_hot = load_molecule_xyz(file, dataset_info)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        dists = torch.cdist(positions.unsqueeze(0),
                            positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]
        # print("Average distance between atoms", dists.mean().item())
        plot_data3d(positions, atom_type, dataset_info=dataset_info,
                    save_path=file[:-4] + '.png',
                    spheres_3d=spheres_3d)

        if wandb is not None:
            path = file[:-4] + '.png'
            # Log image(s)
            im = plt.imread(path)
            wandb.log({'molecule': [wandb.Image(im, caption=path)]})


def visualize_chain(path, dataset_info, wandb=None, spheres_3d=False,
                    mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        file = files[i]

        positions, one_hot = load_molecule_xyz(file, dataset_info=dataset_info)

        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, dataset_info=dataset_info,
                    save_path=fn, spheres_3d=spheres_3d, alpha=1.0)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


def visualize_chain_uncertainty(
        path, dataset_info, wandb=None, spheres_3d=False, mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        if i + 2 == len(files):
            break

        file = files[i]
        file2 = files[i + 1]
        file3 = files[i + 2]

        positions, one_hot, _ = load_molecule_xyz(file,
                                                  dataset_info=dataset_info)
        positions2, one_hot2, _ = load_molecule_xyz(
            file2, dataset_info=dataset_info)
        positions3, one_hot3, _ = load_molecule_xyz(
            file3, dataset_info=dataset_info)

        all_positions = torch.stack([positions, positions2, positions3], dim=0)
        one_hot = torch.stack([one_hot, one_hot2, one_hot3], dim=0)

        all_atom_type = torch.argmax(one_hot, dim=2).numpy()
        fn = file[:-4] + '.png'
        plot_data3d_uncertainty(
            all_positions, all_atom_type, dataset_info=dataset_info,
            save_path=fn, spheres_3d=spheres_3d, alpha=0.5)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


if __name__ == '__main__':
    # plot_grid()
    import qm9.dataset as dataset
    from configs.datasets_config import qm9_with_h, geom_with_h

    matplotlib.use('macosx')

    task = "visualize_molecules"
    task_dataset = 'geom'

    if task_dataset == 'qm9':
        dataset_info = qm9_with_h


        class Args:
            batch_size = 1
            num_workers = 0
            filter_n_atoms = None
            datadir = 'qm9/temp'
            dataset = 'qm9'
            remove_h = False


        cfg = Args()

        dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

        for i, data in enumerate(dataloaders['train']):
            positions = data['positions'].view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            plot_data3d(
                positions_centered, atom_type, dataset_info=dataset_info,
                spheres_3d=True)

    elif task_dataset == 'geom':
        files = load_xyz_files('outputs/data')
        matplotlib.use('macosx')
        for file in files:
            x, one_hot, _ = load_molecule_xyz(file, dataset_info=geom_with_h)

            positions = x.view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = one_hot.view(-1, 16).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            mask = (x == 0).sum(1) != 3
            positions_centered = positions_centered[mask]
            atom_type = atom_type[mask]

            plot_data3d(
                positions_centered, atom_type, dataset_info=geom_with_h,
                spheres_3d=False)

    else:
        raise ValueError(dataset)
