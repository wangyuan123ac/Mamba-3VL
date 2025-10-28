import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from modules.utils import get_pairwise_distance

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x.clone()
        x = self.w_1(x) 
        x = self.act(x)
        x = self.dropout1(x)
        x = self.w_2(x) 
        x = self.dropout2(x) 
        return self.ln(x + residual)

class MultiHeadSpatialNet(nn.Module):
    def __init__(self, view_num, n_head=8, d_model=768, d_hidden=2048, dropout=0.1):
        super(MultiHeadSpatialNet, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.view_num = view_num

        self.k_proj= nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        # Create a linear layer for each head
        self.score_layers = nn.ModuleList([nn.Linear(self.d_head, 1) for _ in range(n_head)])
        self.ffn = FeedForwardLayer(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def forward(self, spatial_features):
        BVN, N, _ = spatial_features.shape
        k = self.k_proj(spatial_features).view(BVN, N, self.n_head, self.d_head)
        v = self.v_proj(spatial_features).view(BVN, N, self.n_head, self.d_head)        
        # Initialize containers for the multihead outputs
        multihead_features = []
        multihead_score = []
        for i in range(self.n_head):
            k_head = k[:, :, i, :]
            v_head = v[:, :, i, :]
            score = self.score_layers[i](k_head).squeeze(-1)
            score = F.softmax(score, dim=-1).unsqueeze(-1)
            feature = (score * v_head).sum(dim=1)
            multihead_features.append(feature)
            multihead_score.append(score)
        # Concatenate results from all heads
        feature = torch.cat(multihead_features, dim=-1)
        score = torch.cat(multihead_score, dim=-1).mean(dim=-1)        
        features = self.ffn(feature)
        return features

def create_classifier(d_model, output_dim, dropout_rate):
    return nn.Sequential(nn.Linear(d_model, d_model//3), 
                        nn.ReLU(), 
                        nn.Dropout(dropout_rate), 
                        nn.LayerNorm(d_model//3),
                        nn.Linear(d_model//3, output_dim))
    
def create_mapping(input_dim, output_dim, dropout_rate):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim))

@torch.no_grad()
def aug_input(input_points, contrast_range=(0.5, 1.5), noise_std_dev=0.02, rotate_number=4, training=True, device='cudda'):
    input_points = input_points.float().to(device)
    xyz = input_points[:, :, :, :3]  # Get x,y,z coordinates (B, N, P, 3)
    B, N, P = xyz.shape[:3]
    input_points_multiview = []
    rgb = input_points[..., 3:6].clone()
    # Randomly rotate/color_aug if training
    if training:
        rotate_matrix = get_random_rotation_matrix(rotate_number, device)
        xyz = torch.matmul(xyz.reshape(B*N*P, 3), rotate_matrix).reshape(B, N, P, 3)
        rgb = get_augmented_color(rgb, contrast_range, noise_std_dev, device) 
    # multi-view
    for theta in torch.Tensor([i*2.0*np.pi/rotate_number for i in range(rotate_number)]).to(device):  
        rotate_matrix = get_rotation_matrix(theta, device)
        rotated_xyz = torch.matmul(xyz.reshape(B*N*P, 3), rotate_matrix).reshape(B, N, P, 3)
        rotated_input_points = torch.clone(input_points)
        rotated_input_points[..., :3] = rotated_xyz
        rotated_input_points[..., 3:6] = rgb
        input_points_multiview.append(rotated_input_points)
    # Stack list of tensors into a single tensor
    input_points_multiview = torch.stack(input_points_multiview, dim=1)
    return input_points_multiview

@torch.no_grad()
def aug_box(box_infos, rotate_number, training='True', device='cuda'):
    box_infos = box_infos.float().to(device)
    bxyz = box_infos[...,:3] # B,N,3
    B,N = bxyz.shape[:2]
    # bxyz[..., 0] = scale_to_unit_range(bxyz[..., 0]) # normed x
    # bxyz[..., 1] = scale_to_unit_range(bxyz[..., 1]) # normed y
    # bxyz[..., 2] = scale_to_unit_range(bxyz[..., 2]) # normed z
    # Randomly rotate if training
    # 按列操作后拼接
    bxyz_x = scale_to_unit_range(bxyz[..., 0])  # normed x
    bxyz_y = scale_to_unit_range(bxyz[..., 1])  # normed y
    bxyz_z = scale_to_unit_range(bxyz[..., 2])  # normed z

    # 使用 torch.stack 拼接成新张量
    bxyz = torch.stack([bxyz_x, bxyz_y, bxyz_z], dim=-1)
    if training:
        rotate_matrix = get_random_rotation_matrix(rotate_number, device)
        bxyz = torch.matmul(bxyz.reshape(B*N, 3), rotate_matrix).reshape(B,N,3)        
    # multi-view
    bsize = box_infos[...,3:]
    boxs=[]
    for theta in torch.Tensor([i*2.0*np.pi/rotate_number for i in range(rotate_number)]).to(device):
        rotate_matrix = get_rotation_matrix(theta, device)
        rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
        boxs.append(torch.cat([rxyz,bsize],dim=-1))
    boxs=torch.stack(boxs,dim=1)
    return boxs

def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features

def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)

def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch

def get_random_rotation_matrix(rotate_number, device):
    rotate_theta_arr = torch.Tensor([i*2.0*torch.pi/rotate_number for i in range(rotate_number)]).to(device)
    theta = rotate_theta_arr[torch.randint(0, rotate_number, (1,))]
    return get_rotation_matrix(theta, device)

def get_rotation_matrix(theta, device):
    rotate_matrix = torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0.0],
                                [torch.sin(theta), torch.cos(theta),  0.0],
                                [0.0,           0.0,            1.0]]).to(device)
    return rotate_matrix

def get_augmented_color(rgb, contrast_range=(0.5, 1.5), noise_std_dev=0.02, device='cuda'):
    # RGB Augmentation
    contrast_factor = torch.empty(1).uniform_(contrast_range[0], contrast_range[1]).to(device)
    rgb = rgb * contrast_factor
    noise = torch.normal(mean=0., std=noise_std_dev, size=rgb.shape, device=device)
    rgb = rgb + noise
    rgb = torch.clamp(rgb, -1.0, 1.0)
    return rgb

def scale_to_unit_range(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values
    min_x = torch.min(x, dim=-1, keepdim=True).values
    return x / (max_x - min_x + 1e-9)

def norm_output_scores(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    x_standardized = (x - x_mean) / x_std
    return x_standardized
    
def rotation_aggregate(output):
    B, R, N, _, _ = output.shape
    scaled_output = output
    return (scaled_output / R).sum(dim=1)

def batch_expansion(tensor, n):
    return tensor.unsqueeze(1).repeat(1, n, *([1] * (tensor.dim() - 1))).view(tensor.size(0) * n, *tensor.shape[1:])

# def MiKASA(objs, boxes, device='cuda'):
#     """
#     objs: (B, N, D)
#     text: (B, seq_len, D)
#     boxes: (B, N, 7)
#     """        
#     self.device = device
#     B, N, _ = boxes.shape 
#     # Data view
#     boxes = aug_box(boxes, view_num, training, device)
#     boxes = boxes.reshape(B*view_num, N, 7)
#     xyz = boxes[...,:3]

#     # Get relative positions and pos features
#     relative_positions = get_pairwise_distance(xyz).detach()
#     spatial_features = spatial_enc(relative_positions).reshape(B*view_num*N, N, d_model)
#     # Forward
#     # text_view_N = batch_expansion(batch_expansion(text, self.view_num), N)
#     # spatial_features = self.spatial_text(spatial_features, text_view_N)

#     for i in range(n_layer):
#         objs = obj_layers[i](objs, text)
#         objs_view_N = batch_expansion(batch_expansion(objs, view_num), N)
#         spatial_agg = sp_agg[i](spatial_features+objs_view_N)
#         spatial_agg = rotation_aggregate(spatial_agg.reshape(B, view_num, N, d_model))
#         objs = objs + spatial_agg
#     return objs

if __name__ == "__main__":
    sp_agg = MultiHeadSpatialNet(view_num=4)
    sp_agg = sp_agg.cuda()
    A = torch.randn(128, 50, 6).cuda()
    # print(model(A).shape)
    boxes = aug_box(A, rotate_number=4)
    boxes = boxes.reshape(128*4, 50, 6)
    xyz = boxes[...,:3]
    relative_positions = get_pairwise_distance(xyz).detach() # [512, 50, 50, 8]
    spatial_features = torch.randn(128*4*50, 50, 768).cuda()
    objs = torch.randn(128, 50, 768).cuda()
    objs_view_N = batch_expansion(batch_expansion(objs, 4), 50)
    # spatial_agg = spatial_features+objs_view_N
    
    # print(spatial_agg.shape)
    spatial_agg = sp_agg(spatial_features+objs_view_N)
    spatial_agg = rotation_aggregate(spatial_agg.reshape(128, 4, 50, 768))
    objs = objs + spatial_agg
    print(spatial_agg.shape)