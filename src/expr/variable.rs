use super::{CompilerScope, Expr, Id, WrappedExpr};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Shape;
use tensorflow::Status;
use tensorflow::Variable;

pub struct ReadVariable<D: Data> {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) initial_value: WrappedExpr<D>,
    pub(crate) data_type: D,
}

impl<D: Data> Expr<D> for ReadVariable<D> {
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
        self.data_type().dimensions()
    }

    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status> {
        match compiler_scope.get_variable_output(&self.name) {
            Some(output) => {
                ops::read_variable_op(output.clone(), compiler_scope.borrow_scope_mut())
            }
            None => {
                let initial_value_output = compiler_scope.get_output(&self.initial_value)?;

                let variable = Variable::builder()
                    .initial_value(initial_value_output)
                    .data_type(self.data_type.data_type())
                    .shape(self.data_type.shape())
                    .build(compiler_scope.borrow_scope_mut())?;

                let output = variable.output().clone();

                compiler_scope.add_variable(&self.name, variable);

                ops::read_variable_op(output, compiler_scope.borrow_scope_mut())
            }
        }
    }
}
