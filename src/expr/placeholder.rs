use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use std::rc::Rc;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;

#[derive(Clone)]
pub struct Placeholder<D: Data> {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) data_type: D,
}

impl<D: Data + 'static> Placeholder<D> {
    pub fn read(&self) -> Expr<D> {
        Expr(Rc::new(ReadPlaceholderExpr(self.clone())))
    }

    pub fn refer(&self) -> PlaceholderRef<D> {
        PlaceholderRef {
            id: self.id,
            data_type: self.data_type.clone(),
        }
    }
}

pub struct PlaceholderRef<D: Data> {
    pub(crate) id: Id,
    #[allow(dead_code)]
    pub(crate) data_type: D,
}

pub(crate) struct ReadPlaceholderExpr<D: Data>(Placeholder<D>);

impl<D: Data> ExprImpl<D> for ReadPlaceholderExpr<D> {
    fn id(&self) -> Id {
        self.0.id
    }

    fn data_type(&self) -> D {
        self.0.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.0.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.0.data_type.dimensions()
    }

    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status> {
        let operation = ops::Placeholder::new()
            .dtype(self.0.data_type.data_type())
            .shape(self.0.data_type.shape())
            .build(&mut compiler.borrow_scope_mut().with_op_name(&self.0.name))?;

        Ok(CompiledElement::Operation(operation))
    }
}
