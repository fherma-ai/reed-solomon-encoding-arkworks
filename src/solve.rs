//! The message as coefficients, evaluated on the larger domain.
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};

use crate::fherma::{Inputs, Outputs, Point, Tensor};

/// The `i`-th row of a limb tensor as a field element.
///
/// `from_bigint` refuses a value at or above the modulus. The generator
/// reduces before it writes, so a refusal here means the case is not the one
/// this specification describes — better said than folded away.
fn fe(data: &[u64], i: usize, limbs: usize) -> Fr {
    let mut words = [0u64; 4];
    words[..limbs].copy_from_slice(&data[i * limbs..(i + 1) * limbs]);
    Fr::from_bigint(BigInt::new(words)).expect("a field element below the modulus")
}

fn elements(data: &[u64], limbs: usize) -> Vec<Fr> {
    (0..data.len() / limbs).map(|i| fe(data, i, limbs)).collect()
}

fn rows(values: &[Fr], limbs: usize) -> Tensor<u64> {
    let mut data = Vec::with_capacity(values.len() * limbs);
    for v in values {
        data.extend_from_slice(&v.into_bigint().0[..limbs]);
    }
    Tensor { shape: vec![values.len() as i64, limbs as i64], data }
}

fn row(value: Fr, limbs: usize) -> Tensor<u64> {
    Tensor { shape: vec![limbs as i64], data: value.into_bigint().0[..limbs].to_vec() }
}

#[derive(Default)]
pub struct State {
    limbs: usize,
    m: usize,
    /// The root the point names. It belongs to the point now, so it is read
    /// once here instead of on every case.
    root: Fr,
}

pub fn init(p: &Point) -> State {
    let limbs = p.L as usize;
    State { limbs, m: p.M as usize, root: fe(&p.root.data, 0, limbs) }
}

pub fn run(state: &mut State, inp: &Inputs) -> Outputs {
    let (limbs, m) = (state.limbs, state.m);
    let mut coeffs = elements(&inp.message.data, limbs);
    coeffs.resize(m, Fr::from(0u64));
    let domain = Radix2EvaluationDomain::<Fr>::new(m)
        .expect("a power of two within the field's two-adicity");
    assert_eq!(domain.group_gen, state.root,
               "the domain's root is not the one the point names");
    Outputs { codeword: rows(&domain.fft(&coeffs), limbs) }
}

pub fn free(state: State) { drop(state); }
