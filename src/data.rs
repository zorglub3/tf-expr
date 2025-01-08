use tensorflow::DataType;
use std::PhantomData;

pub trait Data {
    fn data_type() -> DataType;
}

struct FloatData {
    phantom: PhantomData<u8>;
}


