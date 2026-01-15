from timm.models.vision_transformer import Block
import torch
import torch.nn as nn
import numpy as np





class Tracker_Model(nn.Module):
    def __init__(self, hidden_dim, num_obs_per_image):
        super().__init__()
        
        self.NUM_MASK_TOKEN = 256
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) # (1, 1, 384)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.image_decoder_obs_pred_projector = nn.Linear(hidden_dim, hidden_dim)
        
        self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, num_obs_per_image + self.NUM_MASK_TOKEN, hidden_dim), requires_grad=False)  # fixed sin-cos embedding #   cls_token is alse passed to the decoder in mae
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(hidden_dim, int(num_obs_per_image**.5), cls_token=False)
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(hidden_dim, int(self.NUM_MASK_TOKEN**.5), cls_token=False)
        image_decoder_position_embedding = np.concatenate((image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0)
        self.image_decoder_position_embedding.data.copy_(torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0))
        
        self.image_decoder = nn.Sequential(
            Block(hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            )
        self.image_decoder_norm = nn.LayerNorm(hidden_dim)
        self.image_decoder_pred = nn.Linear(hidden_dim, 1152)

    def forward(self, track_tokens):

        # track_tokens: [B, 2*num_tokens_per_images, dim]

        B, NUM_OBS_TOKEN, dim = track_tokens.shape
        assert NUM_OBS_TOKEN % 2 == 0

        NUM_OBS_TOKEN_PER_IMAGE = NUM_OBS_TOKEN // 2

        obs_pred_embedding = self.image_decoder_obs_pred_projector(track_tokens.reshape(-1, track_tokens.shape[-1]))
        obs_pred_embedding = obs_pred_embedding.view(B * (NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE), NUM_OBS_TOKEN_PER_IMAGE, dim)
        mask_tokens = self.mask_token.repeat(B * (NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE), self.NUM_MASK_TOKEN, 1)
        image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1)
        image_decoder_input = image_decoder_input + self.image_decoder_position_embedding
        image_decoder_output = self.image_decoder(image_decoder_input) # (224, 205, 384)
        image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :] # (224, 196, 384)
        image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, dim))
        image_pred = self.image_decoder_pred(image_pred_feature)  # (43904, 768)
        # image_pred = image_pred.view(B * S, self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE, self.NUM_MASK_TOKEN, -1)  
        image_pred = image_pred.view(B, NUM_OBS_TOKEN // NUM_OBS_TOKEN_PER_IMAGE, 1, self.NUM_MASK_TOKEN, -1)  # (112, 2, 1, 196, 768)
        return image_pred





def insert_track_attention_mask(attn_mask, prompt_len, in1_len, in1_padding):
    """
    attn_mask: [B, L, L] bool tensor, where True = allowed, False = blocked
    prompt_len: int
    Returns: [B, L + prompt_len, L + prompt_len] bool tensor
    """
    B, L, _ = attn_mask.shape
    device = attn_mask.device
    dtype = torch.bool
    rep_token_len = 2
    
    # assume input1 and input2 are concatenated in attn_mask
    # in1_len: 562, L = 613
    seqlen1 = in1_len - rep_token_len 
    seqlen2 = L - seqlen1 - rep_token_len 
    total_len = seqlen1 + rep_token_len + prompt_len + seqlen2

    # Start with everything blocked
    new_mask = torch.zeros((B, total_len, total_len), dtype=dtype, device=device)

    # 1️⃣ input1 attends to itself
    new_mask[:, :seqlen1, :seqlen1] = attn_mask[:, :seqlen1, :seqlen1]

    # rep_token attends to input1
    new_mask[:, seqlen1:seqlen1+rep_token_len, :seqlen1] = in1_padding[:, None, :seqlen1].expand(B, rep_token_len, seqlen1)  # attend input1
    
    # rep_token attends to itself
    new_mask[:, seqlen1:seqlen1+rep_token_len, seqlen1:seqlen1+rep_token_len] = True

    # 2️⃣ prompt attends to input1 (not prompt or input2 or rep_token)
    new_mask[:, seqlen1+rep_token_len:seqlen1+rep_token_len+prompt_len, :seqlen1] = in1_padding[:, None, :seqlen1].expand(B, prompt_len, seqlen1)  # attend input1

    # 3️⃣ input2 attends to input1 + rep_token + prompt + itself
    new_mask[:, seqlen1+rep_token_len+prompt_len:, :seqlen1] = in1_padding[:, None, :seqlen1].expand(B, seqlen2, seqlen1)  # attend input1
    new_mask[:, seqlen1+rep_token_len+prompt_len:, seqlen1:] = True
    # optional: copy input2 self-attention region from original
    new_mask[:, seqlen1+rep_token_len+prompt_len:, seqlen1+rep_token_len+prompt_len:] = attn_mask[:, seqlen1+rep_token_len:, seqlen1+rep_token_len:]
    return new_mask


def patchify(imgs, patch_size, vision_model):
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
    target_type = vision_model.weight.dtype
    out = vision_model(imgs.to(dtype=target_type))
    out = out.reshape(shape=(out.shape[0], out.shape[1], out.shape[2]*out.shape[3]))
    out = out.permute(0, 2, 1)
    return out

    

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb