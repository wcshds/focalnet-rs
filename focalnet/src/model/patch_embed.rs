use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        LayerNorm, LayerNormConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

pub struct PatchEmbedFeatures<B: Backend> {
    pub features: Tensor<B, 3>,
    pub height: usize,
    pub width: usize,
}

impl<B: Backend> PatchEmbedFeatures<B> {}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    norm: LayerNorm<B>,
    patch_size: [usize; 2],
}

impl<B: Backend> PatchEmbed<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> PatchEmbedFeatures<B> {
        let x = self.proj.forward(input);
        let [_, _, height, width] = x.dims();
        // Shape: [batch_size, height * width, channels]
        let x = x.flatten(2, 3).swap_dims(1, 2);
        let x = self.norm.forward(x);

        PatchEmbedFeatures {
            features: x,
            height,
            width,
        }
    }

    pub fn calculate_patches_resolution(&self, image_size: [usize; 2]) -> [usize; 2] {
        [
            image_size[0] / self.patch_size[0],
            image_size[1] / self.patch_size[1],
        ]
    }
}

#[derive(Config)]
pub struct PatchEmbedConfig {
    #[config(default = "[4, 4]")]
    patch_size: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "96")]
    embed_dimensions: usize,
    #[config(default = "false")]
    use_conv_embed: bool,
    #[config(default = "false")]
    is_stem: bool,
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbed<B> {
        let proj = match self.use_conv_embed {
            true => {
                // if we choose to use conv embedding, then we treat the stem and non-stem differently
                let [kernel_size, padding, stride] = match self.is_stem {
                    true => [[7, 7], [2, 2], [4, 4]],
                    false => [[3, 3], [1, 1], [2, 2]],
                };
                Conv2dConfig::new([self.in_channels, self.embed_dimensions], kernel_size)
                    .with_stride(stride)
                    .with_padding(PaddingConfig2d::Explicit(padding[0], padding[1]))
                    .init(device)
            }
            false => Conv2dConfig::new([self.in_channels, self.embed_dimensions], self.patch_size)
                .with_stride(self.patch_size)
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .init(device),
        };
        let norm = LayerNormConfig::new(self.embed_dimensions).init(device);

        PatchEmbed {
            proj,
            norm,
            patch_size: self.patch_size,
        }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::LibTorch,
        record::{FullPrecisionSettings, PrettyJsonFileRecorder},
    };

    use crate::utils::tensor_wrapper::TensorWrapper;

    use super::*;

    type Backend = LibTorch;

    #[test]
    fn export_json() {
        let device = Default::default();

        let model: PatchEmbed<Backend> = PatchEmbedConfig::new().init(&device);
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        model.save_file("patch_embed.json", &pjr).unwrap();
    }

    #[test]
    fn test_equivalent_to_pytorch() {
        let device = Default::default();
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        let img: Tensor<Backend, 3> = TensorWrapper::load_tensor("tensor.json", &device).into();
        let img_transformed = img.unsqueeze_dim::<4>(0);

        let model: PatchEmbed<Backend> = PatchEmbedConfig::new().init(&device);
        let model = model
            .load_file("./patch_embed.json", &pjr, &device)
            .unwrap();

        let res = model.forward(img_transformed);
        println!("{}", res.features);
        println!("{}", res.height);
        println!("{}", res.width);
    }

    #[test]
    fn delete_when_release() {
        let device = Default::default();
        let a: Tensor<Backend, 4> = Tensor::random(
            [1, 96, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b: Tensor<Backend, 4> = Tensor::random(
            [1, 1, 56, 56],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        println!("{}", a * b);
    }
}
