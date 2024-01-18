use crate::StrError;
use std::convert::TryFrom;

/// Converts number to i32
#[inline]
pub(crate) fn to_i32(num: usize) -> i32 {
    i32::try_from(num).unwrap()
}

/// Converts string to array of bytes (may truncate)
pub fn str_to_bytes(dest: &mut [u8], src: &str) {
    if dest.len() == src.len() {
        dest.copy_from_slice(src.as_bytes());
    } else if dest.len() > src.len() {
        dest[..src.len()].copy_from_slice(src.as_bytes());
    } else {
        dest.copy_from_slice(&src.as_bytes()[..dest.len()]);
    }
}

/// Converts a vector of bytes to string
pub fn bytes_to_string(bytes: Vec<u8>) -> Result<String, StrError> {
    let res = String::from_utf8(bytes).map_err(|_| "cannot convert bytes to UTF-8 string")?;
    Ok(res.trim_end_matches('\0').to_string())
}

/// Converts a vector of bytes to string (replace invalid parts with �)
pub fn bytes_to_string_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim_end_matches('\0').to_string()
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usize_to_i32_works() {
        let m = 2_usize;
        let x = vec![0.0; m];
        let m_i32 = to_i32(x.len());
        assert_eq!(m_i32, 2_i32);
    }

    #[test]
    fn str_to_bytes_works() {
        let mut buf10 = vec![0u8; 10];
        let mut buf13 = vec![0u8; 13];
        let mut buf20 = vec![0u8; 20];

        str_to_bytes(&mut buf10, "123456789abcd");
        str_to_bytes(&mut buf13, "123456789abcd");
        str_to_bytes(&mut buf20, "123456789abcd");

        let msg = String::from_utf8(buf10).unwrap();
        assert_eq!(msg.len(), 10);
        assert_eq!(msg, "123456789a");

        let msg = String::from_utf8(buf13).unwrap();
        assert_eq!(msg.len(), 13);
        assert_eq!(msg, "123456789abcd");

        let msg = String::from_utf8(buf20).unwrap();
        assert_eq!(msg.len(), 20);
        assert_eq!(msg.trim_end_matches('\0'), "123456789abcd");
    }

    #[test]
    fn str_to_bytes_works_emoji() {
        let mut buf10 = vec![0u8; 10];
        let mut buf20 = vec![0u8; 20];

        str_to_bytes(&mut buf10, "123456789abcd 😊");
        str_to_bytes(&mut buf20, "123456789abcd 😊");

        let msg = String::from_utf8(buf10).unwrap();
        assert_eq!(msg.len(), 10);
        assert_eq!(msg, "123456789a");

        let msg = String::from_utf8(buf20).unwrap();
        assert_eq!(msg.len(), 20);
        assert_eq!(msg.trim_end_matches('\0'), "123456789abcd 😊");
    }

    #[test]
    fn bytes_to_string_works() {
        const EXTRA: u8 = 0;

        let sparkle_heart = vec![240, 159, 146, 150];
        let sparkle_heart = bytes_to_string(sparkle_heart).unwrap();
        assert_eq!("💖", sparkle_heart);

        let sparkle_heart = vec![240, 159, 146, 150, EXTRA, EXTRA, EXTRA, EXTRA];
        let sparkle_heart = bytes_to_string(sparkle_heart).unwrap();
        assert_eq!("💖", sparkle_heart);

        let sparkle_heart_wrong = vec![0, 159, 146, 150];
        assert_eq!(bytes_to_string(sparkle_heart_wrong), Err("cannot convert bytes to UTF-8 string"));
    }

    #[test]
    fn bytes_to_string_lossy_works() {
        const EXTRA: u8 = 0;

        let sparkle_heart = &[240, 159, 146, 150];
        let sparkle_heart = bytes_to_string_lossy(sparkle_heart);
        assert_eq!("💖", sparkle_heart);

        let sparkle_heart = &[240, 159, 146, 150, EXTRA, EXTRA, EXTRA, EXTRA];
        let sparkle_heart = bytes_to_string_lossy(sparkle_heart);
        assert_eq!("💖", sparkle_heart);

        let wrong = b"Hello \xF0\x90\x80World";
        let output = bytes_to_string_lossy(wrong);
        assert_eq!("Hello �World", output);
    }
}
