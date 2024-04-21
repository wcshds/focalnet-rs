use burn::{
    config::Config,
    module::Module,
    nn::{
        pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use super::{
    basic_layer::{BasicLayer, BasicLayerConfig},
    patch_embed::{PatchEmbed, PatchEmbedConfig},
};

#[derive(Module, Debug)]
/// Focal Modulation Network (FocalNet)
pub struct FocalNet<B: Backend> {
    patch_embed: PatchEmbed<B>,
    dropout: Dropout,
    layers: Vec<BasicLayer<B>>,
    layernorm: LayerNorm<B>,
    avgpool: AdaptiveAvgPool1d,
    head: Option<Linear<B>>,
}

impl<B: Backend> FocalNet<B> {
    pub fn forward_features(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.patch_embed.forward(input);
        x.features = self.dropout.forward(x.features);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        let x = self.layernorm.forward(x.features);
        let x = self.avgpool.forward(x.swap_dims(1, 2));
        let x = x.flatten(1, 2);

        return x;
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.forward_features(input);

        if let Some(head) = &self.head {
            x = head.forward(x);
        }

        return x;
    }
}

#[derive(Config)]
pub struct FocalNetConfig {
    #[config(default = "[4, 4]")]
    patch_size: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "1000")]
    num_classes: usize,
    #[config(default = "96")]
    embed_dimensions: usize,
    #[config(default = "vec![2, 2, 6, 2]")]
    depths: Vec<usize>,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "0.0")]
    dropout: f64,
    #[config(default = "0.1")]
    droppath: f64,
    #[config(default = "vec![2, 2, 2, 2]")]
    focal_levels: Vec<usize>,
    #[config(default = "vec![3, 3, 3, 3]")]
    focal_windows: Vec<usize>,
    #[config(default = "false")]
    use_conv_embed: bool,
    #[config(default = "false")]
    use_layer_scale: bool,
    #[config(default = "1e-4")]
    layer_scale_value: f64,
    #[config(default = "false")]
    use_post_layernorm: bool,
    #[config(default = "false")]
    use_post_layernorm_in_modulation: bool,
    #[config(default = "false")]
    normalize_modulator: bool,
}

impl FocalNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FocalNet<B> {
        let num_layers = self.depths.len();
        if num_layers != self.focal_levels.len() || num_layers != self.focal_windows.len() {
            panic!("num_layers, focal_levels and num_layers should have same dimensions.");
        }
        let embed_dimensions_vec: Vec<_> = (0..(num_layers))
            .map(|i| self.embed_dimensions * (2usize.pow(i as u32)))
            .collect();
        let num_features = embed_dimensions_vec[num_layers - 1];

        let patch_embed = PatchEmbedConfig::new()
            .with_patch_size(self.patch_size)
            .with_in_channels(self.in_channels)
            .with_embed_dimensions(self.embed_dimensions)
            .with_use_conv_embed(self.use_conv_embed)
            .with_is_stem(true)
            .init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        // stochastic depth decay rule
        let depths_sum = self.depths.iter().sum();
        let num_linspace: Vec<_> = (0..depths_sum)
            .map(|i| self.droppath / (depths_sum - 1) as f64 * i as f64)
            .collect();
        let mut start = 0;
        let mut droppath_rates = Vec::with_capacity(num_layers);
        for &depth in &self.depths {
            let end = start + depth;
            let tmp = num_linspace[start..end].to_vec();
            droppath_rates.push(tmp);
            start = end;
        }

        let mut layers = Vec::with_capacity(num_layers);
        for idx_layer in 0..num_layers {
            let mut config =
                BasicLayerConfig::new(embed_dimensions_vec[idx_layer], self.depths[idx_layer])
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_dropout(self.dropout)
                    .with_droppath(droppath_rates[idx_layer].clone())
                    .with_focal_level(self.focal_levels[idx_layer])
                    .with_focal_window(self.focal_windows[idx_layer])
                    .with_use_layer_scale(self.use_layer_scale)
                    .with_layer_scale_value(self.layer_scale_value)
                    .with_use_post_layernorm(self.use_post_layernorm)
                    .with_use_post_layernorm_in_modulation(self.use_post_layernorm_in_modulation)
                    .with_normalize_modulator(self.normalize_modulator);
            if idx_layer < num_layers - 1 {
                config = config
                    .with_downsample(true)
                    .with_out_dimensions(embed_dimensions_vec[idx_layer + 1])
                    .with_use_conv_embed(self.use_conv_embed);
            } else {
                config = config.with_downsample(false);
            }

            let layer = config.init(device);
            layers.push(layer);
        }

        let layernorm = LayerNormConfig::new(num_features).init(device);
        let avgpool = AdaptiveAvgPool1dConfig::new(1).init();
        let head = if self.num_classes > 0 {
            let linear = LinearConfig::new(num_features, self.num_classes).init(device);
            Some(linear)
        } else {
            None
        };

        FocalNet {
            patch_embed,
            dropout,
            layers,
            layernorm,
            avgpool,
            head,
        }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::LibTorch,
        record::{FullPrecisionSettings, PrettyJsonFileRecorder},
        tensor::Tensor,
    };

    use crate::utils::tensor_wrapper::TensorWrapper;

    use super::*;

    type Backend = LibTorch;

    #[test]
    fn export_json() {
        let device = Default::default();

        let model: FocalNet<Backend> = FocalNetConfig::new().init(&device);
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        model.save_file("template.json", &pjr).unwrap();
    }

    #[test]
    fn test_equivalent_to_pytorch() {
        let device = Default::default();
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        let img: Tensor<Backend, 3> = TensorWrapper::load_tensor("tensor.json", &device).into();
        let img_transformed = img.unsqueeze_dim::<4>(0);

        let model: FocalNet<Backend> = FocalNetConfig::new().init(&device);
        let model = model
            .load_file("./import_weight/focalnet_tiny_srf.json", &pjr, &device)
            .unwrap();
        let res = model.forward(img_transformed);
        println!("{}", res);
        let idx = res.argmax(1);

        println!("{}", idx);
    }
}
