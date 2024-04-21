use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Distribution, Tensor},
};

use super::{
    focal_modulation::{FocalModulation, FocalModulationConfig},
    patch_embed::PatchEmbedFeatures,
};

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Gelu,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let out = self.linear1.forward(input);
        let out = self.activation.forward(out);
        let out = self.dropout.forward(out);
        let out = self.linear2.forward(out);
        let out = self.dropout.forward(out);

        return out;
    }
}

#[derive(Config)]
struct MlpConfig {
    in_features: usize,
    #[config(default = "None")]
    hidden_features: Option<usize>,
    #[config(default = "None")]
    out_features: Option<usize>,
    #[config(default = "0.0")]
    dropout: f64,
}

impl MlpConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        let in_features = self.in_features;
        let hidden_features = match self.hidden_features {
            Some(value) => value,
            None => in_features,
        };
        let out_features = match self.out_features {
            Some(value) => value,
            None => in_features,
        };

        Mlp {
            linear1: LinearConfig::new(in_features, hidden_features).init(device),
            linear2: LinearConfig::new(hidden_features, out_features).init(device),
            activation: Gelu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Clone, Debug)]
struct DropPath {
    prob: f64,
    scale_by_keep: bool,
}

impl DropPath {
    fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if self.prob == 0.0 || !B::ad_enabled() {
            return input;
        }

        let keep_prob = 1.0 - self.prob;
        let mut shape = [1; D];
        shape[0] = input.dims()[0];
        let random_tensor =
            Tensor::random(shape, Distribution::Bernoulli(keep_prob), &input.device());
        let x = input * random_tensor;

        if self.scale_by_keep {
            return x / keep_prob;
        } else {
            return x;
        }
    }
}

#[derive(Config)]
struct DropPathConfig {
    prob: f64,
    #[config(default = "true")]
    scale_by_keep: bool,
}

impl DropPathConfig {
    fn init(&self) -> DropPath {
        DropPath {
            prob: self.prob,
            scale_by_keep: self.scale_by_keep,
        }
    }
}

#[derive(Module, Debug)]
pub struct FocalNetBlock<B: Backend> {
    layernorm1: LayerNorm<B>,
    layernorm2: LayerNorm<B>,
    modulation: FocalModulation<B>,
    drop_path: DropPath,
    mlp: Mlp<B>,
    gamma1: Option<Param<Tensor<B, 1>>>,
    gamma2: Option<Param<Tensor<B, 1>>>,
    use_post_layernorm: bool,
}

impl<B: Backend> FocalNetBlock<B> {
    pub fn forward(&self, input: PatchEmbedFeatures<B>) -> PatchEmbedFeatures<B> {
        let [height, width] = [input.height, input.width];
        let [batch, _, channels] = input.features.dims();
        let shortcut = input.features.clone();

        // Focal Modulation
        let x = if self.use_post_layernorm {
            input.features
        } else {
            self.layernorm1.forward(input.features)
        };
        let x = x.reshape([batch, height, width, channels]);
        let x = self.modulation.forward(x);
        let x = x.reshape([batch, height * width, channels]);
        let x = if !self.use_post_layernorm {
            x
        } else {
            self.layernorm1.forward(x)
        };

        // Feed Forward Network
        // $ x = shortcut + self.drop_path(self.gamma_1 * x)
        let x_scaled_by_gamma1 = match &self.gamma1 {
            Some(gamma1) => x * gamma1.val().unsqueeze_dims(&[0, 1]),
            None => x,
        };
        let x = shortcut + self.drop_path.forward(x_scaled_by_gamma1);
        // $ x = x + self.drop_path(
        // $     self.gamma_2
        // $    * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x)))
        // $ )
        let x_normed = if self.use_post_layernorm {
            self.layernorm2.forward(self.mlp.forward(x.clone()))
        } else {
            self.mlp.forward(self.layernorm2.forward(x.clone()))
        };
        let x_normed_scaled_by_gamma2 = match &self.gamma2 {
            Some(gamma2) => x_normed * gamma2.val().unsqueeze_dims(&[0, 1]),
            None => x_normed,
        };
        let x = x + self.drop_path.forward(x_normed_scaled_by_gamma2);

        PatchEmbedFeatures {
            features: x,
            height,
            width,
        }
    }
}

#[derive(Config)]
pub struct FocalNetBlockConfig {
    dimensions: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "0.0")]
    dropout: f64,
    #[config(default = "0.0")]
    droppath: f64,
    #[config(default = "1")]
    focal_level: usize,
    #[config(default = "3")]
    focal_window: usize,
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

impl FocalNetBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FocalNetBlock<B> {
        let layernorm1 = LayerNormConfig::new(self.dimensions).init(device);
        let layernorm2 = LayerNormConfig::new(self.dimensions).init(device);
        let modulation =
            FocalModulationConfig::new(self.dimensions, self.focal_window, self.focal_level)
                .with_dropout(self.dropout)
                .with_use_post_layernorm_in_modulation(self.use_post_layernorm_in_modulation)
                .with_normalize_modulator(self.normalize_modulator)
                .with_focal_factor(2)
                .with_bias(true)
                .init(device);
        let drop_path = DropPathConfig::new(self.droppath)
            .with_scale_by_keep(true)
            .init();

        let mlp_hidden_dimensions = (self.dimensions as f64 * self.mlp_ratio) as usize;
        let mlp = MlpConfig::new(self.dimensions)
            .with_hidden_features(Some(mlp_hidden_dimensions))
            .with_out_features(Some(self.dimensions))
            .with_dropout(self.dropout)
            .init(device);

        let [gamma1, gamma2] = if self.use_layer_scale {
            [
                Some(Param::from_tensor(Tensor::full(
                    [self.dimensions],
                    self.layer_scale_value,
                    device,
                ))),
                Some(Param::from_tensor(Tensor::full(
                    [self.dimensions],
                    self.layer_scale_value,
                    device,
                ))),
            ]
        } else {
            [None, None]
        };

        FocalNetBlock {
            layernorm1,
            layernorm2,
            modulation,
            drop_path,
            mlp,
            gamma1,
            gamma2,
            use_post_layernorm: self.use_post_layernorm,
        }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::LibTorch,
        record::{FullPrecisionSettings, PrettyJsonFileRecorder},
    };

    use crate::{
        model::patch_embed::{PatchEmbed, PatchEmbedConfig},
        utils::tensor_wrapper::TensorWrapper,
    };

    use super::*;

    type Backend = LibTorch;

    #[test]
    fn export_json() {
        let device = Default::default();

        let model: FocalNetBlock<Backend> = FocalNetBlockConfig::new(96)
            .with_mlp_ratio(4.0)
            .with_dropout(0.0)
            .with_droppath(0.0)
            .with_focal_level(2)
            .with_focal_window(3)
            .with_use_layer_scale(false)
            .with_layer_scale_value(1e-4)
            .with_use_post_layernorm(false)
            .with_use_post_layernorm_in_modulation(false)
            .with_normalize_modulator(false)
            .init(&device);
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        model.save_file("focal_block.json", &pjr).unwrap();
    }

    #[test]
    fn test_equivalent_to_pytorch() {
        let device = Default::default();
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        let img: Tensor<Backend, 3> = TensorWrapper::load_tensor("tensor.json", &device).into();
        let img_transformed = img.unsqueeze_dim::<4>(0);

        let patch_embed: PatchEmbed<Backend> = PatchEmbedConfig::new().init(&device);
        let patch_embed = patch_embed
            .load_file("./patch_embed.json", &pjr, &device)
            .unwrap();
        let focalnet_block: FocalNetBlock<Backend> = FocalNetBlockConfig::new(96)
            .with_mlp_ratio(4.0)
            .with_dropout(0.0)
            .with_droppath(0.0)
            .with_focal_level(2)
            .with_focal_window(3)
            .with_use_layer_scale(false)
            .with_layer_scale_value(1e-4)
            .with_use_post_layernorm(false)
            .with_use_post_layernorm_in_modulation(false)
            .with_normalize_modulator(false)
            .init(&device);
        let focalnet_block = focalnet_block
            .load_file("./focal_block.json", &pjr, &device)
            .unwrap();

        let tmp = patch_embed.forward(img_transformed);
        let res = focalnet_block.forward(tmp);

        println!("{}", res.features);
    }
}
