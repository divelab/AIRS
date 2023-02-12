conf = {}




# ## skip600000_hidden32_numinter6_lr1e-4_wd1e-5_wbindingsite_cutoff10_bs16
conf_model = {}
conf_model['cutoff'] = 10.0
conf_model['num_node_types'] = 46 # lig_types + rec_types
conf_model['num_lig_node_types'] = 27 # lig_types
conf_model['num_interactions'] = 6
conf_model['num_filters'] = 32
conf_model['num_gaussians'] = 50
conf_model['hidden_channels'] = 32
conf_model['basis_emb_size'] = 32
conf_model['num_spherical'] = 7
conf_model['num_radial'] = 6
conf_model['num_flow_layers'] = 6
conf_model['deq_coeff'] = 0.9
conf_model['use_gpu'] = True

conf_optim = {'lr': 0.0001, 'weight_decay': 0.00001}

conf['model'] = conf_model
conf['optim'] = conf_optim
conf['verbose'] = 100
conf['batch_size'] = 16
conf['epochs'] = 100
conf['chunk_size'] = 20
conf['num_workers'] = 4

conf['gen_model'] = 'GraphBP'
