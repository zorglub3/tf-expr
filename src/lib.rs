extern crate tensorflow;

pub mod data;
pub mod expr;

use crate::data::Data;
use crate::expr::CompiledElement;
use crate::expr::CompilerScope;
use crate::expr::Expr;
use crate::expr::Id;
use std::collections::HashMap;
use std::ops::Deref;
use tensorflow::Code;
use tensorflow::FetchToken;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;

pub struct RuntimeSession {
    elements: HashMap<Id, CompiledElement>,
    session: Session,
}

impl RuntimeSession {
    pub fn new(compiler_scope: CompilerScope) -> Result<Self, Status> {
        let elements = compiler_scope.elements;
        let scope = compiler_scope.scope;
        let session_options = SessionOptions::new();
        let session = Session::new(&session_options, scope.graph().deref())?;

        Ok(Self { elements, session })
    }

    pub fn fetch_initializers(&self, args: &mut SessionRunArgs) {
        for (_k, v) in &self.elements {
            if let CompiledElement::Variable(variable) = v {
                let _ = args.request_fetch(&variable.initializer(), 0);
            }
        }
    }

    pub fn request_fetch<D: Data>(
        &self,
        args: &mut SessionRunArgs,
        expr: &Expr<D>,
    ) -> Result<FetchToken, Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(CompiledElement::Operation(operation)) => Ok(args.request_fetch(&operation, 0)),
            Some(CompiledElement::Variable(_)) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Expression is not an operation",
            )),
            None => Err(Status::new_set_lossy(
                Code::Unknown,
                "Expression isn't compiled",
            )),
        }
    }

    pub fn session_run_args(&self) -> SessionRunArgs {
        SessionRunArgs::new()
    }

    pub fn run(&self, args: &mut SessionRunArgs) -> Result<(), Status> {
        self.session.run(args)
    }

    pub fn run_initializers(&self) -> Result<(), Status> {
        let mut args = self.session_run_args();
        self.fetch_initializers(&mut args);
        self.run(&mut args)
    }
}
