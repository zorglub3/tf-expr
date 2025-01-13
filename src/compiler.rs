use crate::data::*;
use crate::expr::VariableRef;
use crate::expr::{Expr, Id};
use std::collections::HashMap;
use tensorflow::Code;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;
use tensorflow::Variable;

pub struct Compiler {
    pub(crate) scope: Scope,
    pub(crate) elements: HashMap<Id, CompiledElement>,
}

impl Compiler {
    pub fn new(scope: Scope) -> Self {
        Self {
            scope,
            elements: HashMap::new(),
        }
    }

    pub fn new_with_root_scope() -> Self {
        Self {
            scope: Scope::new_root_scope(),
            elements: HashMap::new(),
        }
    }

    pub fn borrow_scope_mut(&mut self) -> &mut Scope {
        &mut self.scope
    }

    pub fn compile<const RANK: usize, D: Data<RANK>>(
        &mut self,
        expr: &Expr<RANK, D>,
    ) -> Result<(), Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(_) => Ok(()),
            None => {
                let element = expr.0.make_operation(self)?;
                self.elements.insert(id, element);
                Ok(())
            }
        }
    }

    pub fn get_output<const RANK: usize, D: Data<RANK>>(
        &mut self,
        expr: &Expr<RANK, D>,
    ) -> Result<Output, Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(CompiledElement::Operation(operation)) => Ok(operation.output(0)),
            Some(CompiledElement::Variable(variable)) => Ok(variable.output().clone()),
            Some(CompiledElement::Optimizer(_, _)) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "You can't use the output from an optimizer",
            )),
            None => {
                let element = expr.0.make_operation(self)?;
                let output = element.output()?;
                self.elements.insert(id, element);
                Ok(output)
            }
        }
    }

    pub fn get_operation<const RANK: usize, D: Data<RANK>>(
        &mut self,
        expr: &Expr<RANK, D>,
    ) -> Result<Operation, Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(CompiledElement::Operation(operation)) => Ok(operation.clone()),
            Some(_) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Expression is not an operation",
            )),
            None => {
                let element = expr.0.make_operation(self)?;
                self.elements.insert(id, element.clone());

                match element {
                    CompiledElement::Operation(operation) => Ok(operation),
                    _ => Err(Status::new_set_lossy(
                        Code::InvalidArgument,
                        "Expression is not an operation",
                    )),
                }
            }
        }
    }

    pub fn get_variable<const RANK: usize, D: Data<RANK>>(
        &mut self,
        expr: &Expr<RANK, D>,
    ) -> Result<Variable, Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(CompiledElement::Variable(variable)) => Ok(variable.clone()),
            Some(_) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Expression is not a variable",
            )),
            None => {
                let operation = expr.0.make_operation(self)?;
                self.elements.insert(id, operation.clone());

                match operation {
                    CompiledElement::Variable(variable) => Ok(variable),
                    _ => Err(Status::new_set_lossy(
                        Code::InvalidArgument,
                        "Expression is not an operation",
                    )),
                }
            }
        }
    }

    pub fn variable_by_ref(&mut self, variable_ref: &VariableRef) -> Result<Variable, Status> {
        let id = variable_ref.id;

        match self.elements.get(&id) {
            Some(CompiledElement::Variable(variable)) => Ok(variable.clone()),
            _ => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Expression is not an operation",
            )),
        }
    }
}

#[derive(Clone)]
pub(crate) enum CompiledElement {
    Operation(Operation),
    Variable(Variable),
    Optimizer(Operation, Vec<Variable>),
}

impl CompiledElement {
    pub fn output(&self) -> Result<Output, Status> {
        match self {
            CompiledElement::Operation(operation) => Ok(operation.output(0)),
            CompiledElement::Variable(variable) => Ok(variable.output().clone()),
            CompiledElement::Optimizer(_, _) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Cannot use the output of an optimizer",
            )),
        }
    }
}
