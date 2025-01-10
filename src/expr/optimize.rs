use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use crate::expr::variable::VariableRef;
use crate::tensorflow::train::Optimizer;
use tensorflow::train::AdadeltaOptimizer;
use tensorflow::train::MinimizeOptions;
use tensorflow::Shape;
use tensorflow::Status;

pub(crate) struct AdaDeltaMinimizeExpr<D: Data> {
    pub(crate) id: Id,
    pub(crate) loss: Expr<D>,
    pub(crate) variables: Vec<VariableRef>,
}

impl<D: Data> ExprImpl<NoData> for AdaDeltaMinimizeExpr<D> {
    fn id(&self) -> Id {
        self.id
    }

    fn data_type(&self) -> NoData {
        NoData::new()
    }

    fn shape(&self) -> Shape {
        self.data_type().shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type().dimensions()
    }

    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status> {
        let loss_output = compiler.get_output(&self.loss)?;

        let optimizer = AdadeltaOptimizer::new();
        let mut variables = Vec::new();

        for v in &self.variables {
            variables.push(compiler.variable_by_ref(v)?);
        }

        let (_variables, operation) = optimizer.minimize(
            compiler.borrow_scope_mut(),
            loss_output,
            MinimizeOptions::default().with_variables(&variables),
        )?;

        Ok(CompiledElement::Operation(operation))
    }
}
