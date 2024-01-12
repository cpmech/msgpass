use std::convert::TryFrom;

/// Converts number to i32
#[inline]
pub(crate) fn to_i32(num: usize) -> i32 {
    i32::try_from(num).unwrap()
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::to_i32;

    #[test]
    fn usize_to_i32_works() {
        let m = 2_usize;
        let x = vec![0.0; m];
        let m_i32 = to_i32(x.len());
        assert_eq!(m_i32, 2_i32);
    }
}
