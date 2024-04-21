use burn::{
    module::{Module, Param},
    record::{FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct TensorWrapper<B: Backend, const D: usize> {
    tensor: Param<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> TensorWrapper<B, D> {
    pub fn new(tensor: Tensor<B, D>) -> Self {
        Self {
            tensor: Param::from_tensor(tensor),
        }
    }

    pub fn save_tensor(self, path: &str) {
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        self.save_file(path, &pjr).unwrap();
    }

    pub fn load_tensor(path: &str, device: &B::Device) -> Self {
        let pjr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        let tmp = Self::empty(device);
        tmp.load_file(path, &pjr, device).expect("Failed to load tensor.")
    }

    pub fn to_tensor(self) -> Tensor<B, D> {
        self.tensor.val()
    }

    fn empty(device: &B::Device) -> Self {
        Self {
            tensor: Param::from_tensor(Tensor::empty([1; D], device)),
        }
    }
}

impl<B: Backend, const D: usize> From<Tensor<B, D>> for TensorWrapper<B, D> {
    fn from(tensor: Tensor<B, D>) -> Self {
        Self::new(tensor)
    }
}

impl<B: Backend, const D: usize> From<TensorWrapper<B, D>> for Tensor<B, D> {
    fn from(value: TensorWrapper<B, D>) -> Self {
        value.to_tensor()
    }
}

#[cfg(test)]
mod test {
    use burn::{backend::LibTorch, tensor::Distribution};

    use super::*;

    type Backend = LibTorch;

    #[test]
    fn export_json() {
        let device = Default::default();

        let tmp: Tensor<Backend, 3> =
            Tensor::random([4, 5, 2], Distribution::Normal(0.0, 1.0), &device);
        println!("{}", tmp);
        let tmp: TensorWrapper<Backend, 3> = tmp.into();
        tmp.save_tensor("tensor.json");
    }

    #[test]
    fn load_json() {
        let device = Default::default();

        let tmp: Tensor<Backend, 3> = TensorWrapper::load_tensor("tensor.json", &device).into();
        println!("{}", tmp);
    }
}
