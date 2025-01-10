use super::{CompiledElement, Expr, Id};
use crate::data::*;
use std::collections::HashMap;
use tensorflow::Code;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;
use tensorflow::Variable;

pub struct CompilerScope {
    pub(crate) scope: Scope,
    pub(crate) elements: HashMap<Id, CompiledElement>,
}

impl CompilerScope {
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

    pub fn get_output<D: Data>(&mut self, expr: &Expr<D>) -> Result<Output, Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(CompiledElement::Operation(operation)) => Ok(operation.output(0)),
            Some(CompiledElement::Variable(variable)) => Ok(variable.output().clone()),
            None => {
                let element = expr.0.make_operation(self)?;
                let output = element.output();
                self.elements.insert(id, element);
                Ok(output)
            }
        }
    }

    pub fn get_operation<D: Data>(&mut self, expr: &Expr<D>) -> Result<Operation, Status> {
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

    pub fn get_variable<D: Data>(&mut self, expr: &Expr<D>) -> Result<Variable, Status> {
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
}
