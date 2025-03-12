import warnings
from collections import OrderedDict

from e3nn.o3 import Irreps

import hienet._const as _const
import hienet._keys as KEY
from hienet.nn.convolution import IrrepsConvolution
import hienet.util as util
from hienet.nn.edge_embedding import (
    BesselBasis,
    ExpNormalSmearing,
    EdgeEmbedding,
    EdgePreprocess,
    PolynomialCutoff,
    SphericalEncoding,
    XPLORCutoff,
    ComformerEdgeEmbedding,
    RBFExpansion
)
from hienet.nn.equivariant_gate import EquivariantGate
from hienet.nn.force_output import ForceOutputFromEdge, ForceStressOutput
from hienet.nn.linear import AtomReduce, FCN_e3nn, IrrepsLinear, get_linear, IrrepsDropoutLinear
from hienet.nn.node_embedding import OnehotEmbedding, CGCNNEmbedding
from hienet.nn.scale import Rescale, SpeciesWiseRescale
from hienet.nn.self_connection import (
    SelfConnectionIntro,
    SelfConnectionLinearIntro,
    SelfConnectionOutro,
)
from hienet.nn.iComfLayers import ComformerNodeConvLayer, ComformerConvEdgeLayer, TensorProductConvLayer, eComfEquivariantConvLayer
from hienet.nn.sequential import AtomGraphSequential
import torch.nn as nn

# warning from PyTorch, about e3nn type annotations
warnings.filterwarnings(
    'ignore',
    message=(
        "The TorchScript type system doesn't "
        "support instance-level annotations"
    ),
)


def init_self_connection(config):
    self_connection_type = config[KEY.SELF_CONNECTION_TYPE]
    intro, outro = None, None
    if self_connection_type == 'none':
        pass
    elif self_connection_type == 'nequip':
        intro, outro = SelfConnectionIntro, SelfConnectionOutro
        return SelfConnectionIntro, SelfConnectionOutro
    elif self_connection_type == 'linear':
        intro, outro = SelfConnectionLinearIntro, SelfConnectionOutro
    else:
        raise ValueError('something went wrong...')
    return intro, outro


def init_radial_basis(config):
    radial_basis_dct = config[KEY.RADIAL_BASIS]
    param = {}
    param.update(radial_basis_dct)
    del param[KEY.RADIAL_BASIS_NAME]

    if radial_basis_dct[KEY.RADIAL_BASIS_NAME] == 'bessel':
        param.update({"cutoff_length": config[KEY.CUTOFF]})
        basis_function =  BesselBasis(**param)
        return basis_function, basis_function.num_basis

    elif radial_basis_dct[KEY.RADIAL_BASIS_NAME] == 'rbf':
        basis_function = RBFExpansion(**param)
        return basis_function, basis_function.n_bins

    elif radial_basis_dct[KEY.RADIAL_BASIS_NAME] == 'orbit_rbf':
        basis_function = ExpNormalSmearing(**param)
        return basis_function, basis_function.num_rbf
    raise RuntimeError('something went very wrong...')


def init_cutoff_function(config):
    cutoff_function_dct = config[KEY.CUTOFF_FUNCTION]
    param = {"cutoff_length": config[KEY.CUTOFF]}
    param.update(cutoff_function_dct)
    del param[KEY.CUTOFF_FUNCTION_NAME]

    if cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'poly_cut':
        return PolynomialCutoff(**param)
    elif cutoff_function_dct[KEY.CUTOFF_FUNCTION_NAME] == 'XPLOR':
        return XPLORCutoff(**param)
    raise RuntimeError('something went very wrong...')


# TODO: it gets bigger and bigger. refactor it
def build_E3_equivariant_model(config: dict, parallel=False):
    """
    IDENTICAL to nequip model
    atom embedding is not part of model
    """
    data_key_weight_input = KEY.EDGE_EMBEDDING  # default

    # parameter initialization
    cutoff = config[KEY.CUTOFF]
    num_species = config[KEY.NUM_SPECIES]

    # should be false for even, and true for odd.
    is_parity = config[KEY.IS_PARITY]

    inv_features = config[KEY.INV_FEATURES]
    irrep_seq = config[KEY.IRREPS_MANUAL]
    sh = config[KEY.SH]


    use_edge_conv = config[KEY.USE_EDGE_CONV]
    triplet_features = config[KEY.TRIPLET_FEATURES]

    lmax = config[KEY.LMAX]

    lmax_edge = (
        config[KEY.LMAX_EDGE]
        if config[KEY.LMAX_EDGE] >= 0
        else lmax
    )

    lmax_node = (
        config[KEY.LMAX_NODE]
        if config[KEY.LMAX_NODE] >= 0
        else lmax
    )

    num_convolution_layer = config[KEY.NUM_CONVOLUTION]

    irreps_spherical_harm = Irreps.spherical_harmonics(
        lmax_edge, -1 if is_parity else 1
    )
    if parallel:
        layers_list = [OrderedDict() for _ in range(num_convolution_layer)]
        layers_idx = 0
        layers = layers_list[0]
    else:
        layers = OrderedDict()

    sc_intro, sc_outro = init_self_connection(config)

    act_scalar = {}
    act_gate = {}
    for k, v in config[KEY.ACTIVATION_SCARLAR].items():
        act_scalar[k] = _const.ACTIVATION_DICT[k][v]
    for k, v in config[KEY.ACTIVATION_GATE].items():
        act_gate[k] = _const.ACTIVATION_DICT[k][v]
    act_radial = _const.ACTIVATION[config[KEY.ACTIVATION_RADIAL]]

    radial_basis_module, radial_basis_num = init_radial_basis(config)

    #angle_radial_basis_module, angle_radial_basis_num = init_radial_basis(config)
    #triplet_radial_basis_module, triplet_radial_basis_num = init_radial_basis(config)  

    cutoff_function_module = init_cutoff_function(config)

    conv_denominator = config[KEY.CONV_DENOMINATOR]
    if not isinstance(conv_denominator, list):
        conv_denominator = [conv_denominator] * (len(irrep_seq) - 1)

    train_conv_denominator = config[KEY.TRAIN_DENOMINTAOR]

    use_bias_in_linear = config[KEY.USE_BIAS_IN_LINEAR]

    _normalize_sph = config[KEY._NORMALIZE_SPH]


    use_iComf_embedding = config[KEY.USE_COMF_EMBEDDING]


    if use_iComf_embedding:         
        # this can also be tuned
        edge_dim = radial_basis_num
        
        edge_embedding = ComformerEdgeEmbedding(
            basis_module=radial_basis_module,
            radial_basis_num = radial_basis_num,
            out_dim = edge_dim,
            #triplet_features = triplet_features,
            spherical_module = SphericalEncoding(lmax_edge, -1 if is_parity else 1, normalize=_normalize_sph)
        )

    else: 
        edge_dim = radial_basis_num

        edge_embedding = EdgeEmbedding(
            # operate on ||r||
            basis_module=radial_basis_module,
            cutoff_module=cutoff_function_module,
            # operate on r/||r||
            spherical_module=SphericalEncoding(lmax_edge, -1 if is_parity else 1, normalize=_normalize_sph),
            use_edge_conv = use_edge_conv,
            angle_module = nn.Sequential(
                RBFExpansion(
                    r_min=-1.0,
                    r_max=1.0,
                    n_bins=triplet_features,
                ),
                nn.Linear(triplet_features, edge_dim),
                nn.Softplus(),
            ),
    )

    if not parallel:
        layers.update({
            # simple edge preprocessor module with no param
            'edge_preprocess': EdgePreprocess(is_stress=True),
        })

    layers.update({
        # 'Not' simple edge embedding module
        'edge_embedding': edge_embedding,
    })
    # ~~ node embedding to first irreps feature ~~ #
    # here, one hot embedding is preprocess of data not part of model
    # see AtomGraphData._data_for_E3_equivariant_model


    use_CGCNN_embedding = config[KEY.USE_CGCNN_EMBEDDING]
    
    if use_CGCNN_embedding:
        node_embedding_size = 92
    else:
        node_embedding_size = num_species    
        
    one_hot_irreps = Irreps(f'{node_embedding_size}x0e')
    irreps_x = (
        Irreps(f'{feature_multiplicity}x0e')
        if irrep_seq is None
        else Irreps(irrep_seq[0])
    )
    layers.update({
        'onehot_idx_to_onehot': CGCNNEmbedding() if use_CGCNN_embedding else OnehotEmbedding(num_classes=num_species),
        'onehot_to_feature_x': IrrepsDropoutLinear(
            irreps_in=one_hot_irreps,
            irreps_out=Irreps(f'{inv_features[0]}x0e'),
            data_key_in=KEY.NODE_FEATURE,
            dropout = config[KEY.DROPOUT],
            biases=use_bias_in_linear,
        ),
    })

    # Parallel code is not changed
    if parallel:
        layers.update({
            # Do not change its name (or see deploy.py before change)
            'one_hot_ghost': OnehotEmbedding(
                data_key_x=KEY.NODE_FEATURE_GHOST,
                num_classes=num_species,
                data_key_save=None,
                data_key_additional=None,
            ),
            # Do not change its name (or see deploy.py before change)
            'ghost_onehot_to_feature_x': IrrepsLinear(
                irreps_in=one_hot_irreps,
                irreps_out=irreps_x,
                data_key_in=KEY.NODE_FEATURE_GHOST,
                biases=use_bias_in_linear,
            ),
        })

    # ~~ edge feature(convoluiton filter) ~~ #

    # here, we can infer irreps or weight of tp from lmax and f0_irreps
    # get all possible irreps from tp (here we drop l > lmax)
    irreps_node_attr = one_hot_irreps

    weight_nn_hidden = config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS]
    # output layer determined at each IrrepsConvolution layer
    weight_nn_layers = [radial_basis_num] + weight_nn_hidden


    inv_convolutional_layer = config[KEY.NUM_INVARIANT_CONV]
    equiv_convolution_layer = len(irrep_seq) - 1

    for i in range(inv_convolutional_layer):
        # convolution part, l>lmax is droped as defined in irreps_out
        layers.update({f'{i}_invariant_convolution': ComformerNodeConvLayer(
            in_channels=inv_features[i],
            out_channels=inv_features[i+1], 
            heads=1, 
            edge_dim= edge_dim,#irreps_x.dim  
            dropout_mlp = config[KEY.DROPOUT],
            dropout_attn = config[KEY.DROPOUT_ATTN]
            )})
        if(use_edge_conv and i == 0):
           layers.update({f'{i}_edge_convolution': ComformerConvEdgeLayer(
               in_channels=edge_dim,
               out_channels=edge_dim,
               edge_dim=edge_dim
           ) })

    node_features_in = irrep_seq[0]


   
    for i in range(equiv_convolution_layer):


        if i == equiv_convolution_layer - 1:
            lmax_node = 0
            parity_mode = 'even'
        else:
            lmax_node = config[KEY.LMAX] 
            parity_mode = 'full'

        node_features_in = irrep_seq[i]
        node_features_out = irrep_seq[i+1]
            
        layers.update({
            f'{i}_eComf_equiv_convolution': eComfEquivariantConvLayer(
                node_features_in = node_features_in,
                node_features_out = node_features_out,
                edge_dim = edge_dim,
                use_bias_in_linear = use_bias_in_linear,
                act_gate = act_gate,
                act_scalar = act_scalar,
                dropout = config[KEY.DROPOUT_ATTN],
                # Currently uses the same denominator for all layers
                denominator = conv_denominator[0],
                weight_layer_input_to_hidden = weight_nn_layers,
                weight_layer_act = act_radial,
                lmax = lmax_node,
                parity_mode = parity_mode,
                sh = sh
            )

        })

    if config[KEY.USE_DENOISING]:
        layers.update({
            'denoising_head': DenoisingHead(
                data_key_in=KEY.NODE_FEATURE,
                data_key_out=KEY.NODE_FEATURE,
                constant=0.0,
            ),
        })


    if config[KEY.READOUT_AS_FCN] is False:
        mid_dim = (
            feature_multiplicity
            if irrep_seq is None
            else Irreps(irrep_seq[-1]).num_irreps
        )
        hidden_irreps = Irreps([(mid_dim // 2, (0, 1))])
        layers.update({
            'reduce_input_to_hidden': IrrepsDropoutLinear(
            irrep_seq[-1],
            hidden_irreps,
            data_key_in=KEY.NODE_FEATURE,
            biases=use_bias_in_linear,
            dropout=config[KEY.DROPOUT],
            ),
            'reduce_hidden_to_energy': IrrepsLinear(
                hidden_irreps,
                Irreps([(1, (0, 1))]),
                data_key_in=KEY.NODE_FEATURE,
                data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                biases=use_bias_in_linear,
            ),
        })
    else:
        act = _const.ACTIVATION[config[KEY.READOUT_FCN_ACTIVATION]]
        hidden_neurons = config[KEY.READOUT_FCN_HIDDEN_NEURONS]
        layers.update({
            'readout_FCN': FCN_e3nn(
                dim_out=1,
                hidden_neurons=hidden_neurons,
                activation=act,
                data_key_in=KEY.NODE_FEATURE,
                data_key_out=KEY.SCALED_ATOMIC_ENERGY,
                irreps_in=irrep_seq[-1],
            )
        })

    shift = config[KEY.SHIFT]
    scale = config[KEY.SCALE]

    train_shift_scale = config[KEY.TRAIN_SHIFT_SCALE]
    rescale_module = (
        SpeciesWiseRescale
        if config[KEY.USE_SPECIES_WISE_SHIFT_SCALE]
        else Rescale
    )
    layers.update({
        'rescale_atomic_energy': rescale_module(
            shift=shift,
            scale=scale,
            data_key_in=KEY.SCALED_ATOMIC_ENERGY,
            data_key_out=KEY.ATOMIC_ENERGY,
            train_shift_scale=train_shift_scale,
        ),
        'reduce_total_enegy': AtomReduce(
            data_key_in=KEY.ATOMIC_ENERGY,
            data_key_out=KEY.PRED_TOTAL_ENERGY,
            constant=1.0,
        ),
    })


    if not parallel:
        fso = ForceStressOutput(
            data_key_energy=KEY.PRED_TOTAL_ENERGY,
            data_key_force=KEY.PRED_FORCE,
            data_key_stress=KEY.PRED_STRESS,
        )
        fof = ForceOutputFromEdge(
            data_key_energy=KEY.PRED_TOTAL_ENERGY,
            data_key_force=KEY.PRED_FORCE,
        )
        gradient_module = fso if not parallel else fof
        layers.update({'force_output': gradient_module})

    # output extraction part
    type_map = config[KEY.TYPE_MAP]
    if parallel:
        return [AtomGraphSequential(v, cutoff, type_map) for v in layers_list]
    else:
        return AtomGraphSequential(layers, cutoff, type_map)
