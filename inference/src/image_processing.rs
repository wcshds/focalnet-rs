use std::path::Path;

use burn::tensor::{backend::Backend, Data, Shape, Tensor};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug)]
pub struct ImageReader {
    imgs: Vec<u8>,
    batch: usize,
    height: usize,
    width: usize,
}

impl ImageReader {
    pub fn read_images<P: AsRef<Path>>(paths: &[P], height: usize, width: usize) -> ImageReader {
        let batch = paths.len();
        let mut total_img_vec = Vec::with_capacity(batch * height * width * 3);
        for path in paths {
            let img = image::open(path)
                .expect(&format!("Cannot read image from path: {:?}", path.as_ref()));
            let resized_img = img.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::Triangle,
            );

            let mut img_vec = resized_img.into_rgb8().into_vec();
            total_img_vec.append(&mut img_vec);
        }

        Self {
            imgs: total_img_vec,
            height,
            width,
            batch,
        }
    }

    pub fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, 4> {
        let data = Data::new(
            self.imgs,
            Shape::new([self.batch, self.height, self.width, 3]),
        );
        let tensor = Tensor::<B, 4>::from_data(data.convert(), device);
        let tensor = tensor.permute([0, 3, 1, 2]) / 255;
        let tensor = (tensor
            - Tensor::from_floats(&IMAGENET_MEAN[..], device).reshape([1, 3, 1, 1]))
            / Tensor::from_floats(&IMAGENET_STD[..], device).reshape([1, 3, 1, 1]);

        tensor
    }
}
