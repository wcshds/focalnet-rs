use burn::{
    backend::NdArray,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings},
};
use focalnet::model::focal_net::{FocalNet, FocalNetConfig};
use inference::{image_processing::ImageReader, imagenet_classes::IMAGENET};

// focalnet_tiny_srf:
//     DROP_PATH_RATE: 0.2
//     EMBED_DIM: 96
//     DEPTHS: [2, 2, 6, 2]
//     FOCAL_LEVELS: [2, 2, 2, 2]
//     FOCAL_WINDOWS: [3, 3, 3, 3]
// focalnet_tiny_lrf:
//     DROP_PATH_RATE: 0.2
//     EMBED_DIM: 96
//     DEPTHS: [2, 2, 6, 2]
//     FOCAL_LEVELS: [3, 3, 3, 3]
//     FOCAL_WINDOWS: [3, 3, 3, 3]
// focalnet_small_srf:
//     DROP_PATH_RATE: 0.3
//     EMBED_DIM: 96
//     DEPTHS: [2, 2, 18, 2]
//     FOCAL_LEVELS: [2, 2, 2, 2]
//     FOCAL_WINDOWS: [3, 3, 3, 3]
// focalnet_small_lrf:
//     DROP_PATH_RATE: 0.3
//     EMBED_DIM: 96
//     DEPTHS: [2, 2, 18, 2]
//     FOCAL_LEVELS: [3, 3, 3, 3]
//     FOCAL_WINDOWS: [3, 3, 3, 3]
// focalnet_base_srf:
//     DROP_PATH_RATE: 0.5
//     EMBED_DIM: 128
//     DEPTHS: [2, 2, 18, 2]
//     FOCAL_LEVELS: [2, 2, 2, 2]
//     FOCAL_WINDOWS: [3, 3, 3, 3]
// focalnet_base_lrf:
//     DROP_PATH_RATE: 0.5
//     EMBED_DIM: 128
//     DEPTHS: [2, 2, 18, 2]
//     FOCAL_LEVELS: [3, 3, 3, 3]
//     FOCAL_WINDOWS: [3, 3, 3, 3]
fn main() {
    type Backend = NdArray;
    let device = Default::default();
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();

    let paths = [
        "../images/ILSVRC2012_val_00000001_n01751748.JPEG",
        "../images/ILSVRC2012_val_00000002_n09193705.JPEG",
        "../images/ILSVRC2012_val_00000003_n02105855.JPEG",
        "../images/ILSVRC2012_val_00000004_n04263257.JPEG",
        "../images/ILSVRC2012_val_00000005_n03125729.JPEG",
        "../images/ILSVRC2012_val_00000006_n01735189.JPEG",
        "../images/ILSVRC2012_val_00000007_n02346627.JPEG",
        "../images/ILSVRC2012_val_00000008_n02776631.JPEG",
        "../images/ILSVRC2012_val_00000009_n03794056.JPEG",
        "../images/ILSVRC2012_val_00000010_n02328150.JPEG",
        "../images/ILSVRC2012_val_00000011_n01917289.JPEG",
        "../images/ILSVRC2012_val_00000012_n02125311.JPEG",
    ];
    let images = ImageReader::read_images(&paths, 224, 224);
    let images = images.to_tensor(&device);

    let model: FocalNet<Backend> = FocalNetConfig::new()
        .with_droppath(0.5)
        .with_embed_dimensions(128)
        .with_depths(vec![2, 2, 18, 2])
        .with_focal_levels(vec![3, 3, 3, 3])
        .with_focal_windows(vec![3, 3, 3, 3])
        .init(&device);
    let model = model
        .load_file("../pretrained/focalnet_base_lrf.bin", &bfr, &device)
        .unwrap();
    let res = model.forward(images);
    let indices = res.argmax(1).to_data().value;

    for (&pred_idx, path) in indices.iter().zip(paths) {
        let label = &path[(path.len() - 14)..(path.len() - 5)];
        println!(
            "expected: {:?}, pred: {:?}",
            label,
            IMAGENET.get_index(pred_idx as usize).unwrap(),
        );
    }
}
