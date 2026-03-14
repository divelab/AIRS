import torch


def expand_edge_index(edge_index, num_atoms, num_repeats, global_offset, device):
    """
    Expands a set of edges 'num_repeats' times.
    Shifts indices for each repeat so they form disjoint graphs.
    """
    if num_repeats == 0:
        return torch.empty((2, 0), device=device, dtype=torch.long)

    # 1. Create the offsets for each band [0, N, 2N, ...]
    # Shape: (1, num_repeats, 1)
    band_offsets = (torch.arange(num_repeats, device=device) * num_atoms).view(1, -1, 1)
    
    # 2. Expand edges: (2, E) -> (2, 1, E)
    edges_expanded = edge_index.unsqueeze(1)
    
    # 3. Broadcast add: (2, num_repeats, E)
    # This adds the band offset to the node indices
    edges_shifted = edges_expanded + band_offsets
    
    # 4. Flatten to 2D: (2, num_repeats * E)
    edges_flat = edges_shifted.flatten(start_dim=1)
    
    # 5. Add the global accumulated offset (from previous molecules)
    return edges_flat + global_offset


def generate_sampling_mask(num_states, max_samples, num_atoms, device):
    """
    Selects a random subset of bands and creates a boolean mask.
    """
    if max_samples is not None and num_states > max_samples:
        # Randomly sample bands
        num_selected = max_samples
        perm = torch.randperm(num_states, device=device)[:num_selected]
        
        # Create mask for original bands
        mask_mol = torch.zeros(num_states, dtype=torch.bool, device=device)
        mask_mol[perm] = True
    else:
        # Keep all bands
        num_selected = num_states
        mask_mol = torch.ones(num_states, dtype=torch.bool, device=device)

    # Expand mask to cover all atoms in those bands
    # (num_bands) -> (num_bands, num_atoms) -> flatten
    full_mask = mask_mol.unsqueeze(1).repeat(1, num_atoms).flatten()
    
    return full_mask, num_selected


def broadcast_edge_features(global_data, batch_data, max_state_samples, device):
    # --- Setup ---
    global_edge_index = global_data.edge_index
    if global_data.batch.numel() == 0:
        empty_edges = torch.empty((2, 0), device=device, dtype=torch.long)
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        empty_mask = torch.empty((0,), device=device, dtype=torch.bool)
        return {
            "all_eband_edge_index": empty_edges,
            "all_eband_edge_batch": empty_long,
            "sampled_eband_edge_index": empty_edges,
            "sampled_eband_edge_batch": empty_long,
            "sampled_mask": empty_mask,
            "sampled_graph_batch": empty_long,
        }

    global_edge_mol_batch = global_data.batch[global_edge_index[0]]
    edge_inds = torch.arange(global_edge_index.shape[1], device=device)
    batch_size = int(global_data.batch.max().item()) + 1

    # Accumulators
    outputs = {
        "all_edges": [], "all_batch": [], 
        "sampled_edges": [], "sampled_batch": [], 
        "sampled_mask": [], "sampled_graph_batch": []
    }
    
    # Offset trackers
    offsets = {"all": 0, "sampled": 0, "sampled_graph": 0}

    # --- Main Loop ---
    for i in range(batch_size):
        # 1. Extract Molecule Info
        num_states = batch_data['state_data'].num_states[i].item()
        num_atoms = global_data.natoms[i].item()
        
        # Get edges belonging to this molecule
        mol_mask = (global_edge_mol_batch == i)
        mol_edge_index = global_edge_index[:, mol_mask]
        mol_edge_batch_ids = edge_inds[mol_mask]

        # 2. Process "All Bands" Stream
        expanded_edges = expand_edge_index(
            mol_edge_index, num_atoms, num_states, offsets["all"], device
        )
        outputs["all_edges"].append(expanded_edges)
        outputs["all_batch"].append(mol_edge_batch_ids.repeat(num_states))
        
        # Update offset: previous molecules + (current_bands-1 copies)
        offsets["all"] += num_atoms * (num_states - 1)

        # 3. Process "Sampled Bands" Stream
        mask, num_sampled = generate_sampling_mask(num_states, max_state_samples, num_atoms, device)
        outputs["sampled_mask"].append(mask)

        sampled_edges = expand_edge_index(
            mol_edge_index, num_atoms, num_sampled, offsets["sampled"], device
        )
        outputs["sampled_edges"].append(sampled_edges)
        outputs["sampled_batch"].append(mol_edge_batch_ids.repeat(num_sampled))
        
        # Create sampled graph batch vector [0,0,0, 1,1,1...]
        sampled_batch_vec = torch.arange(num_sampled, device=device).unsqueeze(1).repeat(1, num_atoms).flatten()
        outputs["sampled_graph_batch"].append(sampled_batch_vec + offsets["sampled_graph"])
        
        # Update offsets
        offsets["sampled"] += num_atoms * (num_sampled - 1)
        offsets["sampled_graph"] += num_sampled

    # --- Final Concatenation ---
    return {
        "all_eband_edge_index": torch.cat(outputs["all_edges"], dim=1),
        "all_eband_edge_batch": torch.cat(outputs["all_batch"]),
        "sampled_eband_edge_index": torch.cat(outputs["sampled_edges"], dim=1),
        "sampled_eband_edge_batch": torch.cat(outputs["sampled_batch"]),
        "sampled_mask": torch.cat(outputs["sampled_mask"]),
        "sampled_graph_batch": torch.cat(outputs["sampled_graph_batch"]),
    }
