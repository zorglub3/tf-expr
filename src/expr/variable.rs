use super::{CompiledElement, CompilerScope, Expr, Id, WrappedExpr};
use crate::data::*;
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

    fn make_operation(
        &self,
        compiler_scope: &mut CompilerScope,
    ) -> Result<CompiledElement, Status> {
        let initial_value_output = compiler_scope.get_output(&self.initial_value)?;

        let variable = Variable::builder()
            .initial_value(initial_value_output)
            .data_type(self.data_type.data_type())
            .shape(self.data_type.shape())
            .build(&mut compiler_scope.borrow_scope_mut().with_op_name(&self.name))?;

        Ok(CompiledElement::Variable(variable))
    }
}
