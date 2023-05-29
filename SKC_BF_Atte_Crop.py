from turtle import forward
from torch import nn
from torch import cat
import torch
import math
import monai.transforms as transforms

class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)# shape = (batch_size, patch_size ** 2, hidden_size)
        
        return attention_output


class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        
        self.conv1 = convBlock(inplace,outplace,kernel_size=1,padding=0)
        self.conv2 = convBlock(outplace,outplace,kernel_size=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=8):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(features, features, kernel_size=(5+i*2,5+i*2,3), stride=stride, padding=(2+i,2+i,1), groups=G),#kernel_size=3+i*2   padding=1+i
                nn.BatchNorm3d(features),
                nn.ReLU(inplace=False)
            ))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):   
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.stack([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1, keepdim=False)

        fea_s = fea_U.mean(-1).mean(-1).mean(-1)
        fea_z = self.fc(fea_s)         
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.stack([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        fea_v = torch.sum(feas * attention_vectors, dim=1, keepdim=False)

        return fea_v

class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, M=2, G=8, r=2, mid_features=None, stride=1, L=8):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        G = mid_features
        self.feas = nn.Sequential(
            nn.Conv3d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm3d(mid_features),
            SKConv(mid_features, M, G, r, stride=stride, L=L),
            nn.BatchNorm3d(mid_features),
            nn.Conv3d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm3d(out_features)
        )
        if in_features == out_features: 
            self.shortcut = nn.Sequential()
        else: 
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm3d(out_features)
            )
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

class skunet_block_2_3d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.feature = nn.Sequential(
                SKUnit(in_dim, out_dim),
                nn.ReLU(), 
                SKUnit(out_dim, out_dim),
                nn.ReLU(), )
        self.pool = nn.MaxPool3d(2, stride=2)
    
    def forward(self, x):
        conv = self.feature(x)
        x = self.pool(conv)
        return conv, x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=12):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc1 = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc2(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidualFusionBlock(nn.Module):
    def __init__(self, in_planes, ratio = 16, kernel_size = 3):
        super(ResidualFusionBlock, self).__init__()
        self.channel_atte = ChannelAttention(in_planes, ratio)
        self.spatial_atte = SpatialAttention(kernel_size)
        self.conv1 = nn.Conv3d(in_planes, in_planes, 1, bias=False)
        self.conv2 = nn.Sequential(nn.Conv3d(in_planes, in_planes, 3, bias=False, padding=1),
                                nn.BatchNorm3d(in_planes),
                                nn.ReLU(),
                                nn.Conv3d(in_planes, in_planes, 3, bias=False, padding=1))
    
    def forward(self, x):
        out = self.channel_atte(x) * x
        out = self.spatial_atte(out) * out
        out = self.conv1(x + out)
        out = self.conv2(out) + out
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(True)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu1(x0)
        x = self.conv2(x0)
        x = self.bn2(x)
        x = self.relu2(x) + x0
        out = self.pool(x)
        return x, out

class Down2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(True)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out = self.pool(x)
        return x, out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.sample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(True)

    def forward(self, x, conv):
        x = self.sample(x) 
        x = cat((x, conv), dim=1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class SKC_BF_Atte_Crop(nn.Module):
    def __init__(self, in_channels=1, num_filters=8, class_num=3, dropout_rate=0.5, atte_drop_rate=0.5, nblock=2, slice_num=48, use_gender=False):
        super().__init__()

        self.nblock_end = nblock
        self.down1 = skunet_block_2_3d(in_channels, num_filters)
        self.down2 = skunet_block_2_3d(num_filters, num_filters * 2)
        self.down3 = skunet_block_2_3d(num_filters * 2, num_filters * 4)
        self.down4 = skunet_block_2_3d(num_filters * 4, num_filters * 8)

        self.bridge = nn.Conv3d(num_filters * 8, num_filters * 16, 3, padding=1)

        self.skip1 = ResidualFusionBlock(num_filters)
        self.skip2 = ResidualFusionBlock(num_filters * 2)
        self.skip3 = ResidualFusionBlock(num_filters * 4)
        self.skip4 = ResidualFusionBlock(num_filters * 8)
        self.skip5 = ResidualFusionBlock(num_filters * 16)

        self.up1 = Up(num_filters * 16, num_filters * 8)
        self.up2 = Up(num_filters * 8, num_filters * 4)
        self.up3 = Up(num_filters * 4, num_filters * 2)
        self.up4 = Up(num_filters * 2, num_filters)

        self.conv_class = nn.Conv3d(num_filters, class_num, 1)
        self.end_down1 = Down2(num_filters, num_filters * 2)
        self.end_down2 = Down2(num_filters * 2, num_filters * 8)
        self.end_down3 = Down2(num_filters * 8, num_filters * 16)

        self.slice_num = slice_num
        self.avg = nn.AdaptiveAvgPool3d(1)
        if use_gender:
            self.glo_dense = nn.Linear(num_filters * 16 + 4, num_filters * 4)
        else:
            self.glo_dense = nn.Linear(num_filters * 16 + 3, num_filters * 4)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.glo_dense2 = nn.Linear(num_filters * 4, num_filters * 1)
        self.glo_dense3 = nn.Linear(num_filters, 1)

        self.attnlistend = nn.ModuleList()
        self.fftlistend = nn.ModuleList()
        
        for n in range(self.nblock_end):
            atten = GlobalAttention(
                    transformer_num_heads=8,
                    hidden_size=num_filters*16,
                    transformer_dropout_rate=atte_drop_rate)
            self.attnlistend.append(atten)
            
            fft = Feedforward(inplace=num_filters*32, outplace=num_filters*16)
            self.fftlistend.append(fft)

    def forward(self, x, gender):

        conv1, x = self.down1(x)
        conv1 = self.skip1(conv1)
        conv2, x = self.down2(x)
        conv2 = self.skip2(conv2)
        conv3, x = self.down3(x)
        conv3 = self.skip3(conv3)
        conv4, x = self.down4(x)
        conv4 = self.skip4(conv4)
        x = self.bridge(x)
        x = self.skip5(x)

        glo_out = x
        B1,C1,L1,H1,W1 = glo_out.size()

        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)
        seg_out = self.conv_class(x)

        x = x.detach()
        seg = torch.argmax(seg_out.detach(), dim = 1, keepdim = True)
        B0,C0,L0,H0,W0 = seg.size()
        whole_size = L0 * H0 * W0
        DGM_size = (L0 // 2, H0 // 2, W0 // 2)
        WM_size = torch.sum(torch.flatten((seg == 8), start_dim = 2), dim = -1)
        GM_size = torch.sum(torch.flatten((seg == 7), start_dim = 2), dim = -1)
        CSF_size = torch.sum(torch.flatten((seg == 6), start_dim = 2), dim = -1)
        WM_ratio = WM_size / whole_size
        GM_ratio = GM_size / whole_size
        CSF_ratio = CSF_size / whole_size
        DGM_mask = (seg > 0) * (seg < 6)
        DGM_area = []
        for i in range(B0):
            volume_current, mask_current = x[i], DGM_mask[i]
            data = {"image":volume_current, "label":mask_current}
            data = transforms.CropForegroundd(keys=("image", "label"), source_key="label", margin=5, mode="reflect")(data)
            volume_feature = data["image"]
            volume_feature = transforms.Resize(spatial_size=DGM_size, mode="trilinear", align_corners=True)(volume_feature)
            DGM_area.append(volume_feature)
        x = torch.stack(DGM_area, dim = 0)
        _, x = self.end_down1(x)
        _, x = self.end_down2(x)
        _, x = self.end_down3(x)

        B,C,L,H,W = x.size()
        x = x.view(B,C,L*H*W).permute(0,2,1).contiguous()
        for n in range(self.nblock_end):

            glo_out_t = glo_out.view(B1,C1,L1*H1*W1).permute(0,2,1)
            tmp = self.attnlistend[n](glo_out_t, x)
            tmp = tmp.permute(0,2,1)
            tmp = tmp.view(B,C,L1,H1,W1)
            tmp = torch.cat([tmp, glo_out], 1)
                    
            tmp = self.fftlistend[n](tmp)
            glo_out = tmp + glo_out
        
        glo_out = torch.flatten(self.avg(glo_out), 1)
        if gender is not None:
            glo_out = torch.cat([glo_out, WM_ratio, GM_ratio, CSF_ratio, gender], dim = 1)
        else:
            glo_out = torch.cat([glo_out, WM_ratio, GM_ratio, CSF_ratio], dim = 1)
        glo_out = self.glo_dense(glo_out)
        glo_out = self.dropout(glo_out)
        glo_out = self.glo_dense2(glo_out)
        glo_out = self.glo_dense3(glo_out)
        
        return seg_out, glo_out