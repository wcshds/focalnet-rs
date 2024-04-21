use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::burn_ext::sequential::Sequential;

#[derive(Module, Debug)]
pub struct FocalModulation<B: Backend> {
    pre_linear: Linear<B>,            // self.f
    mix_channel: Conv2d<B>,           // self.h
    activation: Gelu,                 // self.act
    post_linear: Linear<B>,           // self.proj
    dropout: Dropout,                 // self.proj_drop
    focal_layers: Vec<Sequential<B>>, // self.focal_layers
    layernorm: Option<LayerNorm<B>>,  // self.ln
    normalize_modulator: bool,
}

impl<B: Backend> FocalModulation<B> {
    // Input tensor's Shape: [batch, height, width, dimensions]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, height, width, channels] = input.dims();
        let focal_level = self.focal_layers.len();

        // pre linear projection
        // shape: [batch, channels, height, width]
        let x = self.pre_linear.forward(input).permute([0, 3, 1, 2]);
        // torch.split
        let query = x
            .clone()
            .slice([0..batch, 0..channels, 0..height, 0..width]);
        let mut context =
            x.clone()
                .slice([0..batch, channels..(channels * 2), 0..height, 0..width]);
        let gates = x.slice([
            0..batch,
            (channels * 2)..(channels * 2 + focal_level + 1),
            0..height,
            0..width,
        ]);

        // context aggreation
        let mut context_sum = context.zeros_like();
        for level in 0..focal_level {
            // hierarchical contextualization
            context = self.focal_layers[level].forward(context);
            // gated aggregation
            let gated_aggreation = context.clone()
                * gates
                    .clone()
                    .slice([0..batch, level..(level + 1), 0..height, 0..width]);
            context_sum = context_sum + gated_aggreation;
        }
        let context_global = self.activation.forward(context.mean_dim(2).mean_dim(3));
        context_sum = context_sum
            + context_global
                * gates.slice([
                    0..batch,
                    focal_level..(focal_level + 1),
                    0..height,
                    0..width,
                ]);

        // normalize context
        if self.normalize_modulator {
            context_sum = context_sum / ((focal_level + 1) as f32)
        }

        // focal modulation
        let modulator = self.mix_channel.forward(context_sum);
        let mut out = query * modulator;
        out = out.permute([0, 2, 3, 1]);
        if let Some(layernorm) = &self.layernorm {
            out = layernorm.forward(out);
        }

        // post linear porjection
        out = self.post_linear.forward(out);
        out = self.dropout.forward(out);

        out
    }
}

#[derive(Config)]
pub struct FocalModulationConfig {
    dimensions: usize,
    focal_window: usize,
    focal_level: usize,
    #[config(default = "2")]
    focal_factor: usize,
    #[config(default = "true")]
    bias: bool,
    #[config(default = "0.0")]
    dropout: f64,
    #[config(default = "false")]
    use_post_layernorm_in_modulation: bool,
    #[config(default = "false")]
    normalize_modulator: bool,
}

impl FocalModulationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FocalModulation<B> {
        let pre_linear =
            LinearConfig::new(self.dimensions, self.dimensions * 2 + self.focal_level + 1)
                .with_bias(self.bias)
                .init(device);
        let mix_channel = Conv2dConfig::new([self.dimensions, self.dimensions], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(self.bias)
            .init(device);
        let activation = Gelu::new();
        let post_linear = LinearConfig::new(self.dimensions, self.dimensions)
            .with_bias(self.bias)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        let mut focal_layers = Vec::with_capacity(self.focal_level);
        for level in 0..(self.focal_level) {
            let kernel_size = self.focal_factor * level + self.focal_window;
            let layer_vec = vec![
                Conv2dConfig::new(
                    [self.dimensions, self.dimensions],
                    [kernel_size, kernel_size],
                )
                .with_groups(self.dimensions)
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(kernel_size / 2, kernel_size / 2))
                .with_bias(false)
                .init(device)
                .into(),
                Gelu::new().into(),
            ];
            let layer = Sequential::from(layer_vec);
            focal_layers.push(layer);
        }
        let layernorm = if self.use_post_layernorm_in_modulation {
            Some(LayerNormConfig::new(self.dimensions).init(device))
        } else {
            None
        };

        FocalModulation {
            pre_linear,
            mix_channel,
            activation,
            post_linear,
            dropout,
            focal_layers,
            layernorm,
            normalize_modulator: self.normalize_modulator,
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
    fn test_equivalent_to_pytorch() {
        let device = Default::default();
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        let img: Tensor<Backend, 3> = TensorWrapper::load_tensor("tensor.json", &device).into();
        let img_transformed = img.unsqueeze_dim::<4>(0);

        let patch_embed: PatchEmbed<Backend> = PatchEmbedConfig::new().init(&device);
        let patch_embed = patch_embed
            .load_file("./patch_embed.json", &pjr, &device)
            .unwrap();
        let focal_modulation = FocalModulationConfig::new(96, 3, 2)
            .with_focal_factor(2)
            .with_bias(true)
            .with_dropout(0.0)
            .with_use_post_layernorm_in_modulation(false)
            .with_normalize_modulator(false)
            .init(&device);
        let focal_modulation = focal_modulation
            .load_file("./focal_modulation.json", &pjr, &device)
            .unwrap();

        let tmp = patch_embed.forward(img_transformed);
        println!("{}", tmp.features);
        println!("{}", tmp.height);
        println!("{}", tmp.width);
        let res = focal_modulation.forward(tmp.features.reshape([1, 56, 56, 96]));
        println!("{}", res);

        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        focal_modulation
            .save_file("focal_modulation.json", &pjr)
            .unwrap();
    }
}
