use crate::not_zero::NotZero::Two;

#[repr(u8)]
pub enum NotZero {
    One(u8) = 1,
    Two([u8; 7]) = 2
}

impl NotZero {
    pub fn two(val: u64) -> Self {
        assert_eq!(val >> (64-8), 0);
        let val = val.to_ne_bytes();
        if cfg!(target_endian = "big") {
            Two(*val.last_chunk().unwrap())
        } else {
            Two(*val.first_chunk().unwrap())
        }
    }

    pub fn val(self) -> u64 {
        match self {
            NotZero::One(x) => x as u64,
            Two(val) => {
                let mut val64 = [0u8;8];
                if cfg!(target_endian = "big") {
                    *val64.last_chunk_mut().unwrap() = val;
                } else {
                    *val64.first_chunk_mut().unwrap() = val;
                }
                u64::from_ne_bytes(val64)
            }
        }
    }
}