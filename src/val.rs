use tensorflow::DataType;
use tensorflow::Operation;
use tensorflow::Status;
use tensorflow::Scope;
use tensorflow::Shape;
use tensorflow::ops;
use tensorflow::Code;
use crate::expr::Expr;

#[derive(Debug)]
pub enum Value<'a> {
    RandomNormal(DataType, Shape),
    RandomPoisson(Box<Expr<'a>>, DataType, Shape),
    Constant(f64, DataType, Shape),
}

impl<'a> PartialEq for Value<'a> {
    fn eq(&self, other: &Value<'a>) -> bool {
        todo!()
    }
}

impl<'a> Value<'a> {
    pub fn to_operation(&self, scope: &mut Scope) -> Result<Operation, Status> {
        use Value::*;

        match self {
            RandomNormal(data_type, shape) => {
                ops::RandomStandardNormal::new()
                    .dtype(*data_type)
                    .build(shape_operation(shape, scope)?, scope)
            }
            RandomPoisson(rate, data_type, shape) => {
                ops::RandomPoisson::new()
                    .dtype(*data_type)
                    .build(shape_operation(shape, scope)?, rate.to_operation(scope)?, scope)
            }
            Constant(v, data_type, shape) => todo!(),
        }
    }
}

fn shape_operation(shape: &Shape, scope: &mut Scope) -> Result<Operation, Status> {
    let mut shape_values = Vec::new();

    let Some(dims) = shape.dims() else {
        return Err(Status::new_set_lossy(Code::Unknown, "Unknown shape size"));
    };

    for index in 0 .. dims {
        let Some(value) = shape[index] else {
            return Err(Status::new_set_lossy(Code::Unknown, "Unknown shape element"));
        };

        shape_values.push(value);
    }

    ops::constant(&shape_values[..], scope)
}
