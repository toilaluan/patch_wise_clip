import torch

img_patch = torch.randn(64, 49, 768)
txt_latent = torch.randn(1000, 49, 768)

print("img_patch shape:", img_patch.shape)
print("txt_latent shape:", txt_latent.shape)

try:
    # Note: Removed the .sum(-1) as discussed in the previous answer,
    # as the einsum should already produce (B, K)
    patch_logits = torch.einsum("bnd,knd->bk", img_patch, txt_latent)
    print("Einsum worked!")
    print("patch_logits shape:", patch_logits.shape)
except RuntimeError as e:
    print("Error during einsum:")
    print(e)