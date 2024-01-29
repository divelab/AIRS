conf = {}

conf_data = {}
conf_data['prop_name'] = 'formation_energy_per_atom'
conf_data['graph_method'] = 'crystalnn'

conf_model = {}

conf_enc_backbone = {}
conf_enc_backbone['cutoff'] = 5.0
conf_enc_backbone['num_layers'] = 4
conf_enc_backbone['hidden_channels'] = 128
conf_enc_backbone['out_channels'] = 256
conf_enc_backbone['int_emb_size'] = 64
conf_enc_backbone['basis_emb_size_dist'] = 8
conf_enc_backbone['basis_emb_size_angle'] = 8
conf_enc_backbone['basis_emb_size_torsion'] = 8
conf_enc_backbone['out_emb_channels'] = 256
conf_enc_backbone['num_spherical'] = 7
conf_enc_backbone['num_radial'] = 6

conf_dec_backbone = {}
conf_dec_backbone['cutoff'] = 5.0
conf_dec_backbone['num_layers'] = 4
conf_dec_backbone['hidden_channels'] = 128

conf_dec_backbone['out_channels'] = 256
conf_dec_backbone['int_emb_size'] = 64
conf_dec_backbone['basis_emb_size_dist'] = 8
conf_dec_backbone['basis_emb_size_angle'] = 8
conf_dec_backbone['basis_emb_size_torsion'] = 8
conf_dec_backbone['out_emb_channels'] = 256
conf_dec_backbone['num_spherical'] = 7
conf_dec_backbone['num_radial'] = 6

conf_model['enc_backbone_params'] = conf_enc_backbone
conf_model['dec_backbone_params'] = conf_dec_backbone

conf_model['latent_dim'] = 128
conf_model['num_fc_hidden_layers'] = 1
conf_model['fc_hidden_dim'] = 256
conf_model['max_num_atoms'] = 20
conf_model['max_atomic_num'] = 100
conf_model['use_gpu'] = True
conf_model['lattice_scale'] = True
conf_model['pred_prop'] = False
conf_model['use_multi_latent'] = True
conf_model['logvar_clip'] = 6.0
conf_model['mu_clip'] = 14.0
conf_model['num_time_steps'] = 50
conf_model['noise_start'] = 0.01
conf_model['noise_end'] = 10
conf_model['cutoff'] = 5.0
conf_model['max_num_neighbors'] = 5
conf_model['coord_loss_type'] = 'per_node'


conf_optim = {'lr': 0.001, 'betas': [0.9, 0.999], 'weight_decay': 0.0}

conf['kld_weight'] = 0.01
conf['elem_type_num_weight'] = 1.0
conf['elem_type_weight'] = 30.0
conf['elem_num_weight'] = 1.0
conf['lattice_weight'] = 10.0
conf['coord_weight'] = 10.0
conf['max_grad_value'] = 0.5

conf['data'] = conf_data
conf['model'] = conf_model
conf['optim'] = conf_optim
conf['verbose'] = 1
conf['batch_size'] = 256
conf['start_epoch'] = 0
conf['end_epoch'] = 300
conf['save_interval'] = 50
conf['chunk_size'] = 1000
conf['train_temp'] = [1.0, 1.0, 1.0]
conf['gen_temp'] = [0.2, 0.7, 0.7, 0.01]
conf['val_temp'] = [1.0, 1.0, 1.0]
conf['loss_thre'] = [3.0, 10.0]