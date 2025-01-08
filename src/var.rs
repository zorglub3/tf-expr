use tensorflow::DataType;
use tensorflow::Shape;
use tensorflow::Scope;
use tensorflow::Status;
use tensorflow::Output;
use crate::expr::Expr;

/*
pub fn double_variable<'a, S: Into<Shape>>(
    name: &str, 
    shape: S,
    initial_value: &Expr<'a>,
    scope: &mut Scope,
) -> Result<PersistentVariable, Status> {
    PersistentVariable::build_with_initial_value(
        name, 
        DataType::Double, 
        shape, 
        initial_value,
        scope,
    )
}
*/

pub fn variable<'a, S: Into<Shape>>(
    name: &str, 
    data_type: DataType,
    shape: S,
    initial_value: &Expr<'a>,
    scope: &mut Scope,
) -> Result<PersistentVariable, Status> {
    PersistentVariable::build_with_initial_value(
        name, 
        data_type,
        shape, 
        initial_value,
        scope,
    )
}

#[derive(Debug)]
pub struct PersistentVariable {
    name: String,
    data_type: DataType,
    shape: Shape,
    variable: tensorflow::Variable,
}

impl PartialEq for PersistentVariable {
    fn eq(&self, other: &PersistentVariable) -> bool {
        self.name == other.name && self.data_type == other.data_type && self.shape == other.shape
    }
}

impl PersistentVariable {
    pub fn build<S: Into<Shape>>(
        name: &str,
        data_type: DataType,
        shape: S,
        scope: &mut Scope,
    ) -> Result<Self, Status> {
        let shape = shape.into();
        let variable = tensorflow::Variable::builder()
            .data_type(data_type)
            .shape(shape.clone())
            .build(scope)?;

        Ok(Self { 
            name: name.to_string(), 
            data_type,
            shape,
            variable,
        })
    }

    pub fn build_with_initial_value<'a, S: Into<Shape>>(
        name: &str,
        data_type: DataType,
        shape: S,
        initial_value: &Expr<'a>, 
        scope: &mut Scope,
    ) -> Result<Self, Status> {
        let shape = shape.into();
        let variable = tensorflow::Variable::builder()
            .initial_value(initial_value.to_operation(scope)?)
            .data_type(data_type)
            .shape(shape.clone())
            .build(scope)?;

        Ok(Self {
            name: name.to_string(),
            data_type,
            shape,
            variable,
        })
    }

    pub fn output(&self) -> &Output {
        self.variable.output()
    }

    pub fn read(&self) -> Expr {
        Expr::Variable(self)
    }
}
