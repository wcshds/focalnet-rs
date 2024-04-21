use burn::{config::Config, module::Module, tensor::backend::Backend};

use super::{
    focal_block::{FocalNetBlock, FocalNetBlockConfig},
    patch_embed::{PatchEmbed, PatchEmbedConfig, PatchEmbedFeatures},
};

#[derive(Module, Debug)]
pub struct BasicLayer<B: Backend> {
    blocks: Vec<FocalNetBlock<B>>,
    downsample: Option<PatchEmbed<B>>,
}

impl<B: Backend> BasicLayer<B> {
    pub fn forward(&self, input: PatchEmbedFeatures<B>) -> PatchEmbedFeatures<B> {
        let mut x = input;
        for block in &self.blocks {
            x = block.forward(x);
        }

        if let Some(downsample) = &self.downsample {
            let [batch, _, channels] = x.features.dims();
            let [height, width] = [x.height, x.width];
            let tmp = x
                .features
                .swap_dims(1, 2)
                .reshape([batch, channels, height, width]);

            x = downsample.forward(tmp);
        }

        x
    }
}

#[derive(Config)]
pub struct BasicLayerConfig {
    dimensions: usize,
    depth: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "0.0")]
    dropout: f64,
    #[config(default = "vec![0.0]")]
    droppath: Vec<f64>,
    #[config(default = "1")]
    focal_level: usize,
    #[config(default = "1")]
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
    #[config(default = "false")]
    downsample: bool,
    #[config(default = "96")]
    out_dimensions: usize,
    #[config(default = "false")]
    use_conv_embed: bool,
}

impl BasicLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BasicLayer<B> {
        let mut blocks = Vec::with_capacity(self.depth);
        for idx in 0..(self.depth) {
            let droppath = match self.droppath.len() {
                0 => panic!("No droppath provided"),
                1 => self.droppath[0],
                value if value >= self.depth => self.droppath[idx],
                others => panic!("The length of the dropout is less than the required length: expected {} but got {}", self.depth, others)
            };
            let focalnet_block: FocalNetBlock<B> = FocalNetBlockConfig::new(self.dimensions)
                .with_mlp_ratio(self.mlp_ratio)
                .with_dropout(self.dropout)
                .with_droppath(droppath)
                .with_focal_level(self.focal_level)
                .with_focal_window(self.focal_window)
                .with_use_layer_scale(self.use_layer_scale)
                .with_layer_scale_value(self.layer_scale_value)
                .with_use_post_layernorm(self.use_post_layernorm)
                .with_use_post_layernorm_in_modulation(self.use_post_layernorm_in_modulation)
                .with_normalize_modulator(self.normalize_modulator)
                .init(device);
            blocks.push(focalnet_block);
        }

        let downsample = if self.downsample {
            let downsample = PatchEmbedConfig::new()
                .with_patch_size([2, 2])
                .with_in_channels(self.dimensions)
                .with_embed_dimensions(self.out_dimensions)
                .with_use_conv_embed(self.use_conv_embed)
                .with_is_stem(false)
                .init(device);

            Some(downsample)
        } else {
            None
        };

        BasicLayer { blocks, downsample }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::LibTorch,
        record::{FullPrecisionSettings, PrettyJsonFileRecorder},
        tensor::Tensor,
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

        let model: BasicLayer<Backend> = BasicLayerConfig::new(96, 2)
            .with_mlp_ratio(4.0)
            .with_dropout(0.0)
            .with_droppath(vec![0.0, 0.00909090880304575])
            .with_focal_level(2)
            .with_focal_window(3)
            .with_use_layer_scale(false)
            .with_layer_scale_value(1e-4)
            .with_use_post_layernorm(false)
            .with_use_post_layernorm_in_modulation(false)
            .with_normalize_modulator(false)
            .with_downsample(true)
            .with_out_dimensions(192)
            .with_use_conv_embed(false)
            .init(&device);
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        model.save_file("basic_layer.json", &pjr).unwrap();
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
        let basic_layer: BasicLayer<Backend> = BasicLayerConfig::new(96, 2)
            .with_mlp_ratio(4.0)
            .with_dropout(0.0)
            .with_droppath(vec![0.0, 0.00909090880304575])
            .with_focal_level(2)
            .with_focal_window(3)
            .with_use_layer_scale(false)
            .with_layer_scale_value(1e-4)
            .with_use_post_layernorm(false)
            .with_use_post_layernorm_in_modulation(false)
            .with_normalize_modulator(false)
            .with_downsample(true)
            .with_out_dimensions(192)
            .with_use_conv_embed(false)
            .init(&device);
        let basic_layer = basic_layer
            .load_file("./basic_layer.json", &pjr, &device)
            .unwrap();

        let tmp = patch_embed.forward(img_transformed);
        let res = basic_layer.forward(tmp);

        println!("{}", res.features);
        println!("{}", res.height);
        println!("{}", res.width);
    }
}
