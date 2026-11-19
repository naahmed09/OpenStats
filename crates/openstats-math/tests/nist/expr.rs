//! Parser and evaluator for the model formulas in NIST nonlinear-regression
//! files, e.g. `y = b1*(1-exp[-b2*x]) + e`, with forward-mode automatic
//! differentiation so that the fit uses exact Jacobians.

use std::collections::HashMap;

/// Value with its gradient with respect to the parameters.
#[derive(Debug, Clone)]
pub struct Dual {
    pub v: f64,
    pub d: Vec<f64>,
}

impl Dual {
    pub fn constant(v: f64, n: usize) -> Self {
        Dual { v, d: vec![0.0; n] }
    }
    fn map(&self, f: f64, df: f64) -> Dual {
        Dual { v: f, d: self.d.iter().map(|d| df * d).collect() }
    }
    fn zip(&self, o: &Dual, v: f64, da: f64, db: f64) -> Dual {
        Dual { v, d: self.d.iter().zip(&o.d).map(|(x, y)| da * x + db * y).collect() }
    }
    fn is_const(&self) -> bool {
        self.d.iter().all(|d| *d == 0.0)
    }
    fn pow(&self, e: &Dual) -> Dual {
        if e.is_const() {
            let p = e.v;
            let v = if p == p.trunc() && p.abs() < 1e9 { self.v.powi(p as i32) } else { self.v.powf(p) };
            let dv = if p == 0.0 {
                0.0
            } else if p == p.trunc() && p.abs() < 1e9 {
                p * self.v.powi(p as i32 - 1)
            } else {
                p * self.v.powf(p - 1.0)
            };
            return self.map(v, dv);
        }
        let v = self.v.powf(e.v);
        self.zip(e, v, e.v * self.v.powf(e.v - 1.0), v * self.v.ln())
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Call(String, Box<Expr>),
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn expr(&mut self) -> Result<Expr, String> {
        let mut l = self.term()?;
        while let Some(c) = self.peek() {
            if c == b'+' || c == b'-' {
                self.i += 1;
                let r = self.term()?;
                l = Expr::Bin(c as char, Box::new(l), Box::new(r));
            } else {
                break;
            }
        }
        Ok(l)
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        while let Some(c) = self.peek() {
            if c == b'/' || (c == b'*' && self.s.get(self.i + 1) != Some(&b'*')) {
                self.i += 1;
                let r = self.unary()?;
                l = Expr::Bin(c as char, Box::new(l), Box::new(r));
            } else {
                break;
            }
        }
        Ok(l)
    }
    fn unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(b'-') => {
                self.i += 1;
                Ok(Expr::Neg(Box::new(self.unary()?)))
            }
            Some(b'+') => {
                self.i += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }
    fn power(&mut self) -> Result<Expr, String> {
        let base = self.atom()?;
        self.ws();
        let is_pow = self.s.get(self.i) == Some(&b'^')
            || (self.s.get(self.i) == Some(&b'*') && self.s.get(self.i + 1) == Some(&b'*'));
        if is_pow {
            self.i += if self.s[self.i] == b'^' { 1 } else { 2 };
            let e = self.unary()?; // right associative
            return Ok(Expr::Pow(Box::new(base), Box::new(e)));
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Expr, String> {
        let c = self.peek().ok_or("unexpected end of expression")?;
        if c == b'(' || c == b'[' {
            self.i += 1;
            let e = self.expr()?;
            self.close(c)?;
            return Ok(e);
        }
        if c.is_ascii_digit() || c == b'.' {
            let st = self.i;
            while self.i < self.s.len() {
                let ch = self.s[self.i];
                let exp_sign = (ch == b'+' || ch == b'-') && matches!(self.s[self.i - 1], b'e' | b'E') && self.i > st;
                if ch.is_ascii_digit() || ch == b'.' || ch == b'e' || ch == b'E' || exp_sign {
                    // A trailing 'e' that starts an identifier (the model's "+ e") is not part of a number.
                    if (ch == b'e' || ch == b'E') && !matches!(self.s.get(self.i + 1), Some(b'0'..=b'9' | b'+' | b'-'))
                    {
                        break;
                    }
                    self.i += 1;
                } else {
                    break;
                }
            }
            let t = std::str::from_utf8(&self.s[st..self.i]).unwrap();
            return t.parse::<f64>().map(Expr::Num).map_err(|_| format!("bad number {t:?}"));
        }
        if c.is_ascii_alphabetic() {
            let st = self.i;
            while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_') {
                self.i += 1;
            }
            let name = std::str::from_utf8(&self.s[st..self.i]).unwrap().to_ascii_lowercase();
            if matches!(self.peek(), Some(b'(' | b'[')) && is_function(&name) {
                let open = self.s[self.i];
                self.i += 1;
                let arg = self.expr()?;
                self.close(open)?;
                return Ok(Expr::Call(name, Box::new(arg)));
            }
            return Ok(Expr::Var(name));
        }
        Err(format!("unexpected character {:?}", c as char))
    }
    fn close(&mut self, open: u8) -> Result<(), String> {
        let want = if open == b'(' { b')' } else { b']' };
        if self.peek() == Some(want) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("missing {:?}", want as char))
        }
    }
}

fn is_function(n: &str) -> bool {
    matches!(n, "exp" | "log" | "ln" | "log10" | "sin" | "cos" | "tan" | "atan" | "arctan" | "sqrt" | "abs")
}

pub fn parse(src: &str) -> Result<Expr, String> {
    let mut p = P { s: src.as_bytes(), i: 0 };
    let e = p.expr()?;
    if p.peek().is_some() {
        return Err(format!("unexpected trailing input at {:?}", &src[p.i..]));
    }
    Ok(e)
}

impl Expr {
    /// Evaluate with `vars` supplying every identifier (parameters as duals
    /// with unit gradients, predictors as constants).
    pub fn eval(&self, vars: &HashMap<String, Dual>, n: usize) -> Result<Dual, String> {
        Ok(match self {
            Expr::Num(v) => Dual::constant(*v, n),
            Expr::Var(name) => match name.as_str() {
                "pi" => Dual::constant(std::f64::consts::PI, n),
                _ => vars.get(name).cloned().ok_or_else(|| format!("unknown variable {name:?}"))?,
            },
            Expr::Neg(a) => {
                let a = a.eval(vars, n)?;
                a.map(-a.v, -1.0)
            }
            Expr::Bin(op, a, b) => {
                let (a, b) = (a.eval(vars, n)?, b.eval(vars, n)?);
                match op {
                    '+' => a.zip(&b, a.v + b.v, 1.0, 1.0),
                    '-' => a.zip(&b, a.v - b.v, 1.0, -1.0),
                    '*' => a.zip(&b, a.v * b.v, b.v, a.v),
                    _ => a.zip(&b, a.v / b.v, 1.0 / b.v, -a.v / (b.v * b.v)),
                }
            }
            Expr::Pow(a, b) => a.eval(vars, n)?.pow(&b.eval(vars, n)?),
            Expr::Call(f, a) => {
                let a = a.eval(vars, n)?;
                match f.as_str() {
                    "exp" => {
                        let e = a.v.exp();
                        a.map(e, e)
                    }
                    "log" | "ln" => a.map(a.v.ln(), 1.0 / a.v),
                    "log10" => a.map(a.v.log10(), 1.0 / (a.v * std::f64::consts::LN_10)),
                    "sin" => a.map(a.v.sin(), a.v.cos()),
                    "cos" => a.map(a.v.cos(), -a.v.sin()),
                    "tan" => a.map(a.v.tan(), 1.0 / a.v.cos().powi(2)),
                    "atan" | "arctan" => a.map(a.v.atan(), 1.0 / (1.0 + a.v * a.v)),
                    "sqrt" => a.map(a.v.sqrt(), 0.5 / a.v.sqrt()),
                    _ => a.map(a.v.abs(), a.v.signum()),
                }
            }
        })
    }
}
