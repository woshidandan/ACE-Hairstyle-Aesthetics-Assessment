import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from HairEncoder import HairXFormer
from FaceEncoder import FaceXFormer
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from sklearn.cluster import KMeans

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MultiViewFeatureFusion(nn.Module):
    def __init__(self, hair_encoder, num_views=4, feature_dim=256):
        super(MultiViewFeatureFusion, self).__init__()
        self.hair_encoder = hair_encoder
        self.num_views = num_views
        self.view_weights = nn.Parameter(torch.ones(num_views))

        self.angle_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )
        self.angle_feature_extractor = nn.Sequential(
            nn.Linear(12, feature_dim),
            nn.ReLU()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, hair_images):
        B, num_views, C, H, W = hair_images.shape
        hair_images = hair_images.view(B * num_views, C, H, W)
        hair_features = self.hair_encoder(hair_images)
        hair_features = hair_features.view(B, num_views, 3136, -1)
        angle_feature_views = []
        angle_predictions = []
        for i in range(num_views):
            pooled_feature = hair_features[:, i, :, :].mean(dim=1)
            angle_pred = self.angle_predictor(pooled_feature)
            angle_predictions.append(angle_pred)
            angle_feature = self.angle_feature_extractor(angle_pred)
            angle_feature = angle_feature.unsqueeze(1).expand(-1, 3136, -1)
            combined_feature = self.alpha * angle_feature + (1 - self.alpha) * hair_features[:, i, :, :]
            angle_feature_views.append(combined_feature)

        hair_features = torch.stack(angle_feature_views, dim=1)
        weighted_features = hair_features * self.view_weights.view(1, -1, 1, 1)
        fused_features = weighted_features.sum(dim=1) / self.view_weights.sum()
        return fused_features, torch.stack(angle_predictions, dim=1)


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(CrossAttentionModule, self).__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(self, face_feature, hair_feature):
        B, N, F_dim = face_feature.shape
        _, M, H_dim = hair_feature.shape

        cls_face = self.cls_token.expand(B, 1, -1)
        cls_hair = self.cls_token.expand(B, 1, -1)

        face_feature = torch.cat([cls_face, face_feature], dim=1)
        hair_feature = torch.cat([cls_hair, hair_feature], dim=1)

        Q_face = self.query(face_feature)
        K_hair = self.key(hair_feature)
        V_hair = self.value(hair_feature)

        attention_face_to_hair = torch.matmul(Q_face, K_hair.transpose(-2, -1)) / math.sqrt(Q_face.size(-1))
        attention_face_to_hair = F.softmax(attention_face_to_hair, dim=-1)
        output_face_to_hair = torch.matmul(attention_face_to_hair, V_hair)

        K_face = self.key(face_feature)
        V_face = self.value(face_feature)

        attention_hair_to_face = torch.matmul(self.query(hair_feature), K_face.transpose(-2, -1)) / math.sqrt(K_face.size(-1))
        attention_hair_to_face = F.softmax(attention_hair_to_face, dim=-1)
        output_hair_to_face = torch.matmul(attention_hair_to_face, V_face)

        output_face_to_hair_cls = output_face_to_hair[:, 0, :]
        output_hair_to_face_cls = output_hair_to_face[:, 0, :]

        combined_output = torch.cat([output_face_to_hair_cls, output_hair_to_face_cls], dim=-1)

        return combined_output


class AestheticRuleLayer(nn.Module):
    def __init__(self, face_dim=256, hair_dim=256, age_dim=256, color_dim=256):
        super(AestheticRuleLayer, self).__init__()

        self.face_centroids = nn.Parameter(torch.randn(5, face_dim))
        self.hair_centroids = nn.Parameter(torch.randn(12, hair_dim))
        self.age_centroids = nn.Parameter(torch.randn(3, age_dim))
        self.color_centroids = nn.Parameter(torch.randn(4, color_dim))
        self.skin_color_centroids = nn.Parameter(torch.randn(3, color_dim))

        self.rule_matrix_1 = nn.Parameter(torch.zeros(5, 12))
        self.rule_matrix_2 = nn.Parameter(torch.zeros(3, 12))
        self.rule_matrix_3 = nn.Parameter(torch.zeros(3, 4))

        self._init_prior_rules()

    def _init_prior_rules(self):

        prior_matrix_1 = torch.tensor([
            [0.9, 0.2, 0.8, -0.7, 0.5, 0.3, 0.1, -0.2, 0.4, 0.6, 0.7, -0.5],
            [-0.5, 0.8, -0.3, 0.7, 0.2, 0.1, 0.4, 0.3, -0.6, -0.2, 0.3, 0.6],
            [0.2, -0.3, 0.1, -0.5, 0.7, 0.4, 0.2, 0.6, 0.8, 0.3, -0.2, -0.7],
            [0.3, 0.4, -0.2, 0.1, 0.5, 0.7, 0.8, 0.3, -0.4, 0.6, -0.1, 0.2],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ])

        prior_matrix_2 = torch.tensor([
            [0.5, 0.6, 0.7, 0.8, 0.3, 0.6, 0.4, 0.3, 0.5, 0.7, 0.6, 0.8],
            [0.4, 0.3, 0.5, 0.7, 0.6, 0.5, 0.2, 0.3, 0.4, 0.6, 0.7, 0.6],
            [0.3, 0.4, 0.6, 0.5, 0.2, 0.4, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7]
        ])

        prior_matrix_3 = torch.tensor([
            [0.8, 0.2, 0.4, 0.6],
            [0.6, 0.5, 0.7, 0.3],
            [0.4, 0.6, 0.7, 0.8]
        ])

        self.rule_matrix_1.data = prior_matrix_1
        self.rule_matrix_2.data = prior_matrix_2
        self.rule_matrix_3.data = prior_matrix_3

    def update_centroids(self, face_feats, hair_feats, age_feats, color_feats, skin_color_feats):

        with torch.no_grad():
            kmeans = KMeans(n_clusters=5)
            self.face_centroids.data = torch.tensor(kmeans.fit(face_feats.cpu().numpy()).cluster_centers_).to(device)

            kmeans = KMeans(n_clusters=12)
            self.hair_centroids.data = torch.tensor(kmeans.fit(hair_feats.cpu().numpy()).cluster_centers_).to(device)

            kmeans = KMeans(n_clusters=3)
            self.age_centroids.data = torch.tensor(kmeans.fit(age_feats.cpu().numpy()).cluster_centers_).to(device)

            kmeans = KMeans(n_clusters=4)
            self.color_centroids.data = torch.tensor(kmeans.fit(color_feats.cpu().numpy()).cluster_centers_).to(device)

            kmeans = KMeans(n_clusters=3)
            self.skin_color_centroids.data = torch.tensor(kmeans.fit(skin_color_feats.cpu().numpy()).cluster_centers_).to(device)

    def forward(self, face_feat, hair_feat, age_feat, color_feat, skin_color_feat):

        face_sim = F.cosine_similarity(face_feat.unsqueeze(1), self.face_centroids, dim=-1)  # [B, 5]
        hair_sim = F.cosine_similarity(hair_feat.unsqueeze(1), self.hair_centroids, dim=-1)  # [B, 12]
        age_sim = F.cosine_similarity(age_feat.unsqueeze(1), self.age_centroids, dim=-1)  # [B, 3]
        color_sim = F.cosine_similarity(color_feat.unsqueeze(1), self.color_centroids, dim=-1)  # [B, 4]
        skin_color_sim = F.cosine_similarity(skin_color_feat.unsqueeze(1), self.skin_color_centroids, dim=-1)  # [B, 3]

        face_prob = F.softmax(face_sim / 0.1, dim=-1)  # [B, 5]
        hair_prob = F.softmax(hair_sim / 0.1, dim=-1)  # [B, 12]
        age_prob = F.softmax(age_sim / 0.1, dim=-1)  # [B, 3]
        color_prob = F.softmax(color_sim / 0.1, dim=-1)  # [B, 4]
        skin_color_prob = F.softmax(skin_color_sim / 0.1, dim=-1)  # [B, 3]

        rule_score_1 = torch.einsum('bi,ij,bj->b', face_prob, self.rule_matrix_1, hair_prob).unsqueeze(1)
        rule_score_2 = torch.einsum('bi,ij,bj->b', age_prob, self.rule_matrix_2, hair_prob).unsqueeze(1)
        rule_score_3 = torch.einsum('bi,ij,bj->b', skin_color_prob, self.rule_matrix_3, color_prob).unsqueeze(1)

        combined_rule_score = torch.cat([rule_score_1, rule_score_2, rule_score_3], dim=1)

        return combined_rule_score


class HairFaceAestheticModel(nn.Module):
    def __init__(self, hair_encoder_path, face_encoder_path, hair_dim=256, face_dim=256, hidden_dim=512):
        super(HairFaceAestheticModel, self).__init__()

        self.hair_encoder = HairXFormer()
        self.face_encoder = FaceXFormer()
        self.hidden_dim = hidden_dim
        hair_encoder_state_dict = torch.load(
            hair_encoder_path,
            map_location=device,
            weights_only=True,
        )
        self.hair_encoder.load_state_dict(hair_encoder_state_dict, strict=False)

        face_encoder_state_dict = torch.load(
            face_encoder_path,
            map_location=device,
            weights_only=True,
        )
        face_encoder_state_dict = face_encoder_state_dict['state_dict_backbone']
        self.face_encoder.load_state_dict(face_encoder_state_dict, strict=False)

        for param in self.hair_encoder.parameters():
            param.requires_grad = True
        for param in self.face_encoder.parameters():
            param.requires_grad = True

        self.multi_angle_extractor = MultiViewFeatureFusion(self.hair_encoder)
        self.cross_attention = CrossAttentionModule(feature_dim=face_dim, hidden_dim=hidden_dim)

        self.block = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
        num_features = self.block.classifier[2].in_features
        self.block.classifier[2] = nn.Linear(num_features, hidden_dim)

        self.face_convnext_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hair_convnext_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        self.rule_layer = AestheticRuleLayer()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 32),
            nn.ReLU(),
            nn.Linear(hidden_dim // 32, 1),
        )

    def forward(self, hair_images, face_image):
        fused_hair_features, angle_pre = self.multi_angle_extractor(hair_images)
        face_feature_map = self.face_encoder.forward_encoder(face_image)
        cross_attn_features = self.cross_attention(face_feature_map, fused_hair_features)
        _, C = cross_attn_features.shape
        output_face_to_hair_cls = cross_attn_features[:, :C//2]
        output_hair_to_face_cls = cross_attn_features[:, C//2:]
        convnext_features = self.block(face_image)

        face_rule = face_feature_map.mean(dim=1)
        hair_rule = fused_hair_features.mean(dim=1)
        age_rule = face_rule
        color_rule = hair_rule
        skin_color_rule = face_rule
        rule_score = self.rule_layer(face_rule, hair_rule, age_rule, color_rule, skin_color_rule) # [B, 3]

        face_convnext_combined = torch.cat([output_face_to_hair_cls, convnext_features], dim=1)  # [B, hidden_dim * 2]
        face_convnext_fused = self.face_convnext_linear(face_convnext_combined)  # [B, hidden_dim]
        hair_convnext_combined = torch.cat([output_hair_to_face_cls, convnext_features], dim=1)  # [B, hidden_dim * 2]
        hair_convnext_fused = self.hair_convnext_linear(hair_convnext_combined)  # [B, hidden_dim]
        final_combined = torch.cat([face_convnext_fused, hair_convnext_fused, rule_score], dim=1)  # [B, 3 + 2 * hidden_dim]
        score = self.mlp(final_combined).squeeze(-1)  # [B]
        return score, angle_pre
