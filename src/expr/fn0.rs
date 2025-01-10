use super::{ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;

pub(crate) struct Fn0Expr<D: Data> {
    pub(crate) id: Id,
    pub(crate) function: TFFunction0,
    pub(crate) data_type: D,
}

impl<D: Data> ExprImpl<D> for Fn0Expr<D> {
    fn id(&self) -> Id {
        self.id
    }

    fn data_type(&self) -> D {
        self.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type.dimensions()
    }

    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status> {
        let operation = match self.function {
            TFFunction0::RandomStandardNormal => {
                let shape = ops::constant(
                    &self.data_type.dimensions()[..],
                    compiler.borrow_scope_mut(),
                )?;
                ops::random_standard_normal(shape, compiler.borrow_scope_mut())?
            }
            TFFunction0::RandomUniform => {
                let shape = ops::constant(
                    &self.data_type.dimensions()[..],
                    compiler.borrow_scope_mut(),
                )?;
                ops::random_uniform(shape, compiler.borrow_scope_mut())?
            }
        };

        Ok(CompiledElement::Operation(operation))
    }
}

pub(crate) enum TFFunction0 {
    RandomStandardNormal,
    RandomUniform,
}
