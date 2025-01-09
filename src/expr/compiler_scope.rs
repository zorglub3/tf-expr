use super::{Id, WrappedExpr};
use crate::data::*;
use std::collections::HashMap;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;
use tensorflow::Variable;

pub struct CompilerScope {
    scope: Scope,
    operations: HashMap<Id, Operation>,
    variables: HashMap<String, Variable>,
}

impl CompilerScope {
    pub fn new(scope: Scope) -> Self {
        Self {
            scope,
            operations: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    pub fn borrow_scope_mut(&mut self) -> &mut Scope {
        &mut self.scope
    }

    pub fn get_output<D: Data>(&mut self, expr: &WrappedExpr<D>) -> Result<Output, Status> {
        let id = expr.0.id();

        match self.operations.get(&id) {
            Some(operation) => Ok(operation.output(0)),
            None => {
                let operation = expr.0.make_operation(self)?;
                let output = operation.output(0);
                self.operations.insert(id, operation);
                Ok(output)
            }
        }
    }

    pub fn get_operation<D: Data>(&mut self, expr: &WrappedExpr<D>) -> Result<Operation, Status> {
        let id = expr.0.id();

        match self.operations.get(&id) {
            Some(operation) => Ok(operation.clone()),
            None => {
                let operation = expr.0.make_operation(self)?;
                self.operations.insert(id, operation.clone());
                Ok(operation)
            }
        }
    }

    pub fn get_variable_output(&self, name: &str) -> Option<Output> {
        match self.variables.get(name) {
            None => None,
            Some(variable) => Some(variable.output().clone()),
        }
    }

    pub fn add_variable(&mut self, name: &str, variable: Variable) {
        self.variables.insert(name.to_string(), variable);
    }
}
