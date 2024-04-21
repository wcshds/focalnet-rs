use burn::{
    module::Module,
    nn::{conv::Conv2d, pool::MaxPool2d, BatchNorm, Gelu, LayerNorm, Relu},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub enum SequentialType<B: Backend> {
    Conv2d(Conv2d<B>),
    BatchNorm(BatchNorm<B, 2>),
    LayerNorm(LayerNorm<B>),
    Relu(Relu),
    Gelu(Gelu),
    MaxPool2d(MaxPool2d),
}

macro_rules! impl_all_sequential_type {
    ($($type_name:ty, $enum_name:ident);*) => {
        $(
            impl<B: Backend> From<$type_name> for SequentialType<B> {
                fn from(value: $type_name) -> Self {
                    SequentialType::$enum_name(value)
                }
            }
        )*
    };
}

impl_all_sequential_type!(
    Conv2d<B>, Conv2d;
    BatchNorm<B, 2>, BatchNorm;
    LayerNorm<B>, LayerNorm;
    Relu, Relu;
    Gelu, Gelu;
    MaxPool2d, MaxPool2d
);

macro_rules! match_layer {
    ($e1:expr, $e2:expr; $($val:tt),*) => {
        match $e1 {
            $(
                SequentialType::$val(actual_layer) => $e2 = actual_layer.forward($e2),
            )*
        }
    };
}

impl<B: Backend> SequentialType<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        match_layer!(self, x; Conv2d, BatchNorm, LayerNorm, Relu, Gelu, MaxPool2d);

        x
    }
}

#[derive(Module, Debug)]
pub struct Sequential<B: Backend> {
    pub layers: Vec<SequentialType<B>>,
}

impl<B: Backend> Sequential<B> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn from(layers: Vec<SequentialType<B>>) -> Self {
        Self { layers }
    }

    pub fn append(&mut self, layer: SequentialType<B>) {
        self.layers.push(layer);
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;

        for layer in &self.layers {
            x = layer.forward(x);
        }

        x
    }
}

#[cfg(test)]
mod test {
    use burn::backend::NdArray;

    use super::*;

    #[test]
    fn test_sequential() {
        let mut seq = Sequential::<NdArray>::new();

        seq.append(Relu::new().into());
    }
}
