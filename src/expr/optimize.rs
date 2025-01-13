use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use crate::expr::variable::VariableRef;
use crate::tensorflow::train::Optimizer;
use tensorflow::train::AdadeltaOptimizer;
use tensorflow::train::MinimizeOptions;
use tensorflow::Shape;
use tensorflow::Status;

// TODO: GradientDescent, SdcaOptimizer, SdcaOptimizerV2

pub(crate) struct AdaDeltaMinimizeExpr<const RANK: usize, D: Data<RANK>, SD: ScalarData> {
    pub(crate) id: Id,
    pub(crate) loss: Expr<RANK, D>,
    pub(crate) variables: Vec<VariableRef>,
    pub(crate) learning_rate: Option<Expr<0, SD>>,
    pub(crate) rho: Option<Expr<0, SD>>,
    pub(crate) epsilon: Option<Expr<0, SD>>,
}

impl<const RANK: usize, D: Data<RANK>, SD: ScalarData> ExprImpl<0, NoData>
    for AdaDeltaMinimizeExpr<RANK, D, SD>
{
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

        let mut optimizer = AdadeltaOptimizer::new();
        let mut variables = Vec::new();

        for v in &self.variables {
            variables.push(compiler.variable_by_ref(v)?);
        }

        if let Some(learning_rate) = &self.learning_rate {
            let learning_rate_output = compiler.get_output(&learning_rate)?;
            optimizer.set_learning_rate(learning_rate_output);
        }

        if let Some(rho) = &self.rho {
            let rho_output = compiler.get_output(&rho)?;
            optimizer.set_rho(rho_output);
        }

        if let Some(epsilon) = &self.epsilon {
            let epsilon_output = compiler.get_output(&epsilon)?;
            optimizer.set_epsilon(epsilon_output);
        }

        let (variables, operation) = optimizer.minimize(
            compiler.borrow_scope_mut(),
            loss_output,
            MinimizeOptions::default().with_variables(&variables),
        )?;

        Ok(CompiledElement::Optimizer(operation, variables))
    }
}
