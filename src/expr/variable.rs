use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use std::rc::Rc;
use tensorflow::Shape;
use tensorflow::Status;
use tensorflow::Variable as TFVariable;

#[derive(Clone)]
pub struct Variable<D: Data> {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) initial_value: Expr<D>,
    pub(crate) data_type: D,
}

pub struct VariableRef {
    pub(crate) id: Id,
}

impl<D: Data + 'static> Variable<D> {
    pub fn read(&self) -> Expr<D> {
        Expr(Rc::new(ReadVariableExpr(self.clone())))
    }

    pub fn refer(&self) -> VariableRef {
        VariableRef { id: self.id }
    }
}

pub(crate) struct ReadVariableExpr<D: Data>(Variable<D>);

impl<D: Data> ExprImpl<D> for ReadVariableExpr<D> {
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
        let initial_value_output = compiler.get_output(&self.0.initial_value)?;

        let variable = TFVariable::builder()
            .initial_value(initial_value_output)
            .data_type(self.0.data_type.data_type())
            .shape(self.0.data_type.shape())
            .build(&mut compiler.borrow_scope_mut().with_op_name(&self.0.name))?;

        Ok(CompiledElement::Variable(variable))
    }
}
