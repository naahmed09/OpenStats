//! Minimal complex number type (used for eigenvalues of real matrices).

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    pub const fn real(re: f64) -> Self {
        Complex { re, im: 0.0 }
    }
    pub fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
    /// Modulus |z|, computed without overflow.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    pub fn is_real(self) -> bool {
        self.im == 0.0
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl Mul<f64> for Complex {
    type Output = Complex;
    fn mul(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
}
/// Smith's algorithm: avoids overflow/underflow in complex division.
impl Div for Complex {
    type Output = Complex;
    fn div(self, o: Complex) -> Complex {
        if o.im.abs() <= o.re.abs() {
            let r = o.im / o.re;
            let d = o.re + r * o.im;
            Complex::new((self.re + r * self.im) / d, (self.im - r * self.re) / d)
        } else {
            let r = o.re / o.im;
            let d = o.im + r * o.re;
            Complex::new((r * self.re + self.im) / d, (r * self.im - self.re) / d)
        }
    }
}
impl Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}-{}i", self.re, -self.im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        assert_eq!(a * b, Complex::new(11.0, -2.0));
        let q = a / b;
        assert!(((q * b) - a).abs() < 1e-15);
        assert_eq!(a.abs(), 5.0);
        let tiny = Complex::new(1e-300, 1e-300) / Complex::new(1e-300, 0.0);
        assert!((tiny - Complex::new(1.0, 1.0)).abs() < 1e-15);
        assert_eq!(format!("{}", b), "1-2i");
    }
}
