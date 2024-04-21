use std::process::Command;

use burn::{
    backend::NdArray,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
};
use focalnet::model::focal_net::{FocalNet, FocalNetConfig};

fn main() {
    type Backend = NdArray;
    let device = Default::default();
    let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();

    let model: FocalNet<Backend> = FocalNetConfig::new()
        .with_droppath(0.2)
        .with_embed_dimensions(96)
        .with_depths(vec![2, 2, 18, 2])
        .with_focal_levels(vec![3, 3, 3, 3])
        .with_focal_windows(vec![3, 3, 3, 3])
        .init(&device);
    model.clone().save_file("./template.json", &pjr).unwrap();

    // run python script
    run_command();

    let model = model.load_file("./template.json", &pjr, &device).unwrap();
    model
        .save_file("../pretrained/focalnet_small_lrf", &bfr)
        .unwrap();
}

fn run_command() {
    Command::new("python")
        .arg("import_weight_from_pth.py")
        .status()
        .expect("failed to run python script");
}
