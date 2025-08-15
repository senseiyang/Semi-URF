import random
import torch
from typing import Tuple, List


def extract_patches(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Divides a tensor into fixed-size patches.

    Args:
        tensor: Input tensor with shape [B, C, H, W].
        patch_size: The size of the patches.

    Returns:
        patches: A tensor of patches with shape [B, num_patches, patch_size*patch_size].
        num_patches = (H/patch_size) * (W/patch_size).
    """
    B, C, H, W = tensor.shape
    assert C == 1, "Only supports single-channel data."
    assert H % patch_size == 0 and W % patch_size == 0, "H and W should be divisible by patch_size."

    patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Shape (B, C=1, n_h, n_w, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.squeeze(1)  # (B, num_patches, patch_size, patch_size)
    patches = patches.view(B, -1, patch_size * patch_size)  # (B, num_patches, patch_flat)

    return patches


def reconstruct_from_patches(patches: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reconstructs an image from patches, preserving original values.

    Args:
        patches: A tensor of patches with shape [B, num_patches, patch_size*patch_size].
        patch_size: The size of the patches.
        H, W: The height and width of the original image.

    Returns:
        reconstructed: The reconstructed image with shape [B, 1, H, W].
    """
    B, num_patches, _ = patches.shape
    n_h = H // patch_size
    n_w = W // patch_size

    # Reshape patches to [B, n_h, n_w, patch_size, patch_size]
    patches_reshaped = patches.view(B, n_h, n_w, patch_size, patch_size)

    # Create the output tensor
    output = torch.zeros((B, 1, H, W), dtype=patches.dtype, device=patches.device)

    # Use efficient batch operations
    for h in range(n_h):
        h_start = h * patch_size
        h_end = (h + 1) * patch_size

        for w in range(n_w):
            w_start = w * patch_size
            w_end = (w + 1) * patch_size

            # Fill all batch positions at the current location at once
            output[:, 0, h_start:h_end, w_start:w_end] = patches_reshaped[:, h, w]

    return output


def select_patches(mask_patches: torch.Tensor, threshold: float = 0.5) -> List[torch.Tensor]:
    """
    Selects patches based on the proportion of 1s in the mask.
    Args:
        mask_patches: Mask patches with shape [B, num_patches, patch_size*patch_size].
        threshold: Selection threshold, defaults to 0.5.
    Returns:
        selected_indices: A list of selected patch indices for each sample.
                          e.g., [tensor([2, 3]), tensor([1, 3])] means indices 2 and 3 for the first batch,
                          and indices 1 and 3 for the second batch.
    """
    B, n_patches, patch_flat = mask_patches.shape
    selected_indices = []
    thresh_num = patch_flat * threshold

    for i in range(B):
        # Calculate the number of 1s in each patch
        patch_sum = mask_patches[i].sum(dim=1)  # (n_patches,)
        idx = torch.nonzero(patch_sum > thresh_num, as_tuple=False).squeeze(
            1)  # Indices of patches meeting the threshold

        # If no patches meet the condition, select at least one random patch
        if len(idx) == 0:
            idx = torch.tensor([random.randint(0, n_patches - 1)], device=mask_patches.device)

        selected_indices.append(idx)

    return selected_indices  # List[tensor], length equals batch size


def exchange_patches(
        image_l: torch.Tensor,
        gt: torch.Tensor,
        image_un_w: torch.Tensor,
        image_un_s: torch.Tensor,
        mask_l: torch.Tensor,
        mask_unl: torch.Tensor,
        pse_label: torch.Tensor,
        patch_size: int = 64,
        exchange_ratio: float = 0.1,
        random_seed: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements the patch exchange functionality.
    Args:
        image_l: Labeled image, shape [B, 1, H, W].
        gt: Ground truth corresponding to the labeled image, shape [B, 1, H, W].
        image_un_w: Unlabeled weakly augmented image, shape [B, 1, H, W].
        image_un_s: Unlabeled strongly augmented image, shape [B, 1, H, W].
        mask_l: Mask for the labeled image, shape [B, 1, H, W].
        mask_unl: Mask for the unlabeled weakly augmented image, shape [B, 1, H, W].
        pse_label: Pseudo-label for the unlabeled data, shape [B, 1, H, W].
        patch_size: Patch size.
        exchange_ratio: Exchange ratio, defaults to 0.1.
        random_seed: Random seed for reproducible results.

    Returns:
        new_image_l: Updated labeled image.
        new_gt: Updated ground truth.
        new_image_un_w: Updated unlabeled weakly augmented image.
        new_image_un_s: Updated unlabeled strongly augmented image.
        pse_label_temp: Stores ground truth labels of replaced patches, shape [B, 1, H, W], with -1 for non-replaced positions.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        random.seed(random_seed)

    B, C, H, W = image_l.shape
    device = image_l.device

    # 1. Divide into patches
    mask_l_patches = extract_patches(mask_l, patch_size)
    mask_unl_patches = extract_patches(mask_unl, patch_size)

    image_l_patches = extract_patches(image_l, patch_size)
    gt_patches = extract_patches(gt, patch_size)

    image_un_w_patches = extract_patches(image_un_w, patch_size)
    image_un_s_patches = extract_patches(image_un_s, patch_size)
    pse_label_patches = extract_patches(pse_label, patch_size)

    # Total number of patches per sample
    n_patches = mask_l_patches.shape[1]

    # 2. Select patch indices based on masks
    index_l_per_sample = select_patches(mask_l_patches, threshold=0.5)  # List[tensor]
    index_unl_per_sample = select_patches(mask_unl_patches, threshold=0.5)

    # 2.1 Build patch pools
    patch_pool_l = []  # list of (patch, gt_patch, batch_index, patch_idx)
    patch_pool_unl = []  # list of (un_w_patch, un_s_patch, pse_patch, batch_index, patch_idx)

    for b in range(B):
        idx_l = index_l_per_sample[b]
        for pi in idx_l:
            patch_pool_l.append((
                image_l_patches[b, pi].clone(),
                gt_patches[b, pi].clone(),
                b,
                pi.item()
            ))

        idx_unl = index_unl_per_sample[b]
        for pi in idx_unl:
            patch_pool_unl.append((
                image_un_w_patches[b, pi].clone(),
                image_un_s_patches[b, pi].clone(),
                pse_label_patches[b, pi].clone(),
                b,
                pi.item()
            ))

    # Convert to tensor batch format for easier sampling
    if len(patch_pool_l) > 0:
        patch_pool_l_img = torch.stack([p[0] for p in patch_pool_l], dim=0)  # (N_l, patch_dim)
        patch_pool_l_gt = torch.stack([p[1] for p in patch_pool_l], dim=0)
    else:
        patch_pool_l_img = torch.empty((0, patch_size * patch_size), device=device)
        patch_pool_l_gt = torch.empty((0, patch_size * patch_size), device=device)

    if len(patch_pool_unl) > 0:
        patch_pool_unl_w = torch.stack([p[0] for p in patch_pool_unl], dim=0)
        patch_pool_unl_s = torch.stack([p[1] for p in patch_pool_unl], dim=0)
        patch_pool_unl_pse = torch.stack([p[2] for p in patch_pool_unl], dim=0)
    else:
        patch_pool_unl_w = torch.empty((0, patch_size * patch_size), device=device)
        patch_pool_unl_s = torch.empty((0, patch_size * patch_size), device=device)
        patch_pool_unl_pse = torch.empty((0, patch_size * patch_size), device=device)

    # 3. Exchange patches
    # Create copies of the new patch tensors
    new_image_l_patches = image_l_patches.clone()
    new_gt_patches = gt_patches.clone()

    new_image_un_w_patches = image_un_w_patches.clone()
    new_image_un_s_patches = image_un_s_patches.clone()

    # Create new pse_label_temp, initialized with -1 (special value for non-replaced)
    pse_label_temp_patches = torch.full_like(pse_label_patches, fill_value=-1)

    # Record the number of replaced patches (for debugging only)
    replaced_count = 0

    # 3.1 Select patches from the labeled pool to replace in unlabeled images
    for b in range(B):
        idx_unl = index_unl_per_sample[b]
        if len(idx_unl) == 0 or patch_pool_l_img.shape[0] == 0:
            continue

        # Determine the number of samples to exchange
        nsample = min(len(idx_unl), int(len(idx_unl) * exchange_ratio), patch_pool_l_img.shape[0])
        if nsample == 0:
            continue

        # Randomly sample from the labeled pool
        sampled_indices = torch.randperm(patch_pool_l_img.shape[0])[:nsample]
        sampled_patches_img = patch_pool_l_img[sampled_indices]  # (nsample, patch_dim)
        sampled_patches_gt = patch_pool_l_gt[sampled_indices]  # (nsample, patch_dim)

        # Randomly select unlabeled indices for replacement
        replace_indices = torch.randperm(len(idx_unl))[:nsample]
        replace_idx = idx_unl[replace_indices]

        # Replace patches in unlabeled weakly and strongly augmented images
        for k, pi in enumerate(replace_idx):
            new_image_un_w_patches[b, pi] = sampled_patches_img[k]
            new_image_un_s_patches[b, pi] = sampled_patches_img[k]
            pse_label_temp_patches[b, pi] = sampled_patches_gt[k]  # Store GT patches in pse_label_temp
            replaced_count += 1

    # 3.2 Select patches from the unlabeled pool to replace in labeled images
    for b in range(B):
        idx_l = index_l_per_sample[b]
        if len(idx_l) == 0 or patch_pool_unl_w.shape[0] == 0:
            continue

        # Determine the number of samples to exchange
        nsample = min(len(idx_l), int(len(idx_l) * exchange_ratio), patch_pool_unl_w.shape[0])
        if nsample == 0:
            continue

        # Randomly sample from the unlabeled pool
        sampled_indices = torch.randperm(patch_pool_unl_w.shape[0])[:nsample]
        sampled_patches_img = patch_pool_unl_w[sampled_indices]  # (nsample, patch_dim)
        sampled_patches_gt = patch_pool_unl_pse[sampled_indices]  # Pseudo-label patches to be used as replacement GT

        # Randomly select labeled indices for replacement
        replace_indices = torch.randperm(len(idx_l))[:nsample]
        replace_idx = idx_l[replace_indices]

        # Replace patches in the labeled image
        for k, pi in enumerate(replace_idx):
            new_image_l_patches[b, pi] = sampled_patches_img[k]
            new_gt_patches[b, pi] = sampled_patches_gt[k]

    # 4. Reconstruct patches into images
    new_image_l = reconstruct_from_patches(new_image_l_patches, patch_size, H, W)
    new_gt = reconstruct_from_patches(new_gt_patches, patch_size, H, W)

    new_image_un_w = reconstruct_from_patches(new_image_un_w_patches, patch_size, H, W)
    new_image_un_s = reconstruct_from_patches(new_image_un_s_patches, patch_size, H, W)

    # Reconstruct pse_label_temp
    pse_label_temp = reconstruct_from_patches(pse_label_temp_patches, patch_size, H, W)

    # Add debugging information
    # print(f"Number of replaced patches: {replaced_count}/{B*n_patches}")
    return new_image_l, new_gt, new_image_un_w, new_image_un_s, pse_label_temp
