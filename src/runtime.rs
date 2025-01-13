use crate::compiler::CompiledElement;
use crate::compiler::Compiler;
use crate::data::Data;
use crate::expr::Expr;
use crate::expr::Id;
use crate::expr::PlaceholderRef;
use crate::tensordata::TaggedTensor;
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
    pub fn new(compiler: Compiler) -> Result<Self, Status> {
        let elements = compiler.elements;
        let scope = compiler.scope;
        let session_options = SessionOptions::new();
        let session = Session::new(&session_options, scope.graph().deref())?;

        Ok(Self { elements, session })
    }

    pub fn target_initializers(&self, args: &mut SessionRunArgs) -> usize {
        let mut counter = 0_usize;

        for (_k, v) in &self.elements {
            if let CompiledElement::Variable(variable) = v {
                args.add_target(&variable.initializer());
                counter += 1;
            }
        }

        counter
    }

    pub fn request_fetch<const RANK: usize, D: Data<RANK>>(
        &self,
        args: &mut SessionRunArgs,
        expr: &Expr<RANK, D>,
    ) -> Result<FetchToken, Status> {
        let id = expr.0.id();

        match self.elements.get(&id) {
            Some(CompiledElement::Operation(operation)) => Ok(args.request_fetch(&operation, 0)),
            Some(CompiledElement::Variable(_)) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Expression is not an operation",
            )),
            Some(CompiledElement::Optimizer(_, _)) => Err(Status::new_set_lossy(
                Code::InvalidArgument,
                "Cannot fetch output from an optimizer",
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
        let count = self.target_initializers(&mut args);

        if count > 0 {
            self.run(&mut args)
        } else {
            Ok(())
        }
    }

    pub fn add_feed<'l, const RANK: usize, D: Data<RANK> + 'static>(
        &self,
        args: &mut SessionRunArgs<'l>,
        placeholder_ref: &PlaceholderRef<RANK, D>,
        tagged_tensor: &'l TaggedTensor<RANK, D>,
    ) -> Result<(), Status> {
        let id = placeholder_ref.id;

        match self.elements.get(&id) {
            Some(CompiledElement::Operation(operation)) => {
                args.add_feed(&operation, 0, &tagged_tensor.tensor);
                Ok(())
            }
            Some(CompiledElement::Variable(_)) => {
                Err(Status::new_set_lossy(
                    Code::InvalidArgument,
                    "Not a placeholder",
                ))
            }
            Some(CompiledElement::Optimizer(_, _)) => {
                Err(Status::new_set_lossy(
                    Code::InvalidArgument,
                    "Not a placeholder",
                ))
            }
            None => {
                Err(Status::new_set_lossy(
                    Code::Unknown,
                    "Placeholder is not compiled",
                ))
            }
        }
    }
}
