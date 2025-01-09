use tensorflow::Scope;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Status;
use tensorflow::Shape;
use tensorflow::Tensor;
use tensorflow::ops;
use crate::data::*;
use std::rc::Rc;
use std::ops::Deref;
use std::ops::Add;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::collections::HashMap;

static COUNTER: AtomicUsize = AtomicUsize::new(1);

type Id = usize;

fn get_id() -> Id { 
    COUNTER.fetch_add(1, Ordering::Relaxed) 
}

pub trait Expr<D: Data> {
    fn data_type(&self) -> D;
    fn shape(&self) -> Shape;
    fn dimensions(&self) -> Vec<u64>;
    fn id(&self) -> Id;
    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status>;
}

#[derive(Clone)]
pub struct WrappedExpr<D: Data>(Rc<dyn Expr<D>>);

impl<D: Data + 'static> Add<WrappedExpr<D>> for WrappedExpr<D> {
    type Output = WrappedExpr<D>;

    fn add(self, rhs: WrappedExpr<D>) -> WrappedExpr<D> {
        // TODO: make sure dimensions match in `assert_debug!`

        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(BinOpExpr {
            id: get_id(),
            op: BinaryOperator::Add,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<const D: usize> WrappedExpr<FloatData<D>> {
    pub fn tanh(self) -> WrappedExpr<FloatData<D>> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(
            Fn1Expr {
                id: get_id(),
                function: TFFunction::Tanh,
                arg: self.clone(),
                data_type,
            }))
    }

    pub fn exp(self) -> WrappedExpr<FloatData<D>> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(
            Fn1Expr {
                id: get_id(),
                function: TFFunction::Exp,
                arg: self.clone(),
                data_type,
            }))
    }
}

pub struct BinOpExpr<D: Data> {
    id: Id,
    op: BinaryOperator,
    left: WrappedExpr<D>,
    right: WrappedExpr<D>,
    data_type: D,
}

impl<D: Data> Expr<D> for BinOpExpr<D> {
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
        let left_output = compiler_scope.get_output(&self.left)?;
        let right_output = compiler_scope.get_output(&self.right)?;

        match self.op {
            BinaryOperator::Add => ops::add(left_output, right_output, compiler_scope.borrow_scope_mut()),
            BinaryOperator::Sub => ops::sub(left_output, right_output, compiler_scope.borrow_scope_mut()),
            BinaryOperator::Mul => ops::mul(left_output, right_output, compiler_scope.borrow_scope_mut()),
            BinaryOperator::Div => ops::div(left_output, right_output, compiler_scope.borrow_scope_mut()),
        }
    }
}

enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}

pub struct Fn1Expr<D0: Data, D1: Data> {
    id: Id,
    function: TFFunction,
    arg: WrappedExpr<D1>,
    data_type: D0,
}

impl<D0: Data, D1: Data> Expr<D0> for Fn1Expr<D0, D1> {
    fn id(&self) -> Id {
        self.id
    }

    fn data_type(&self) -> D0 {
        self.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type.dimensions()
    }

    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status> {
        let arg_output = compiler_scope.get_output(&self.arg)?;

        match self.function {
            TFFunction::Tanh => ops::tanh(arg_output, compiler_scope.borrow_scope_mut()),
            TFFunction::Exp => ops::exp(arg_output, compiler_scope.borrow_scope_mut()),
        }
    }
}

enum TFFunction {
    Tanh,
    Exp,
}

pub struct ConstantExpr<D: Data> {
    id: Id,
    values: Vec<D::Element>,
    data_type: D,
}

impl<D: Data> Expr<D> for ConstantExpr<D> {
    fn data_type(&self) -> D {
        self.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type.dimensions()
    }

    fn id(&self) -> Id {
        self.id
    }

    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status> {
        let tensor = Tensor::new(&self.data_type().dimensions()[..]).with_values(&self.values)?;
        ops::constant(tensor, compiler_scope.borrow_scope_mut())
    }
}

pub fn scalar_float(v: f32) -> WrappedExpr<FloatData<0>> {
    WrappedExpr(Rc::new(ConstantExpr { 
        id: get_id(),
        values: vec![v],
        data_type: FloatData::from([]),
    }))
}

pub fn vector_float(v: &[f32]) -> WrappedExpr<FloatData<1>> {
    let mut values = Vec::new();

    values.extend_from_slice(v);
    let len = values.len();

    WrappedExpr(Rc::new(ConstantExpr {
        id: get_id(),
        values,
        data_type: FloatData::from([len]),
    }))
}

pub struct CompilerScope {
    scope: Scope,
    operations: HashMap<Id, Operation>,
}

impl CompilerScope {
    pub fn new(scope: Scope) -> Self {
        Self { scope, operations: HashMap::new() }
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
}
