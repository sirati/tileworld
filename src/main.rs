#![feature(new_zeroed_alloc)]
#![feature(generic_const_exprs)]
#![feature(portable_simd)]
#![feature(trivial_bounds)]
#![feature(transparent_unions)]
#![feature(repr_simd)]

use crate::SimdHelper::{MaskInner, SignedPair2};
use crate::SimdHelper::{MaskM, MaskMathable, SimdM, SimdMathable};
use core::simd;
use std::alloc::Layout;
use std::borrow::{Borrow, BorrowMut};
use std::cell::Cell;
use std::convert::Into;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Index, IndexMut};
use std::simd::{mask8x64, u8x64, LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};
use std::simd::prelude::{SimdPartialEq, SimdPartialOrd};
use not_zero::NotZero;
use crate::DataValidity::{Invalid, Unchecked, Valid};
use crate::same_size::{FlattenConstSizeArr, SameAs, SameSizeAs};

// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo   look at todo.md
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 
// todo 

mod not_zero;

#[repr(align(64))]
struct Chunk2dCacheLine([[u8;8];8]);
#[repr(align(4096))] //Index by
struct Chunk2dPageSize([[[Chunk2dCacheLine;4];4];4]);

impl Index<Coord> for Chunk2dPageSize {
    type Output = [Chunk2dCacheLine;4];

    #[inline(always)]
    fn index(&self, index: Coord) -> &Self::Output {
        &(self.0.flatten_ref())[index.region_subindex as usize]
    }
}

impl<'a, Feature: TileWorldFeatureAccessorRaw> Into<&'a FeatureSubRegion<Feature>> for &'a Chunk2dCacheLine {
    fn into(self) -> &'a FeatureSubRegion<Feature> {
        unsafe {
            &*(self as *const Chunk2dCacheLine as *const FeatureSubRegion<Feature>)
        }
    }
}

impl<'a, Feature: TileWorldFeatureAccessorRaw> Into<&'a mut FeatureSubRegion<Feature>> for &'a mut Chunk2dCacheLine {
    fn into(self) -> &'a mut FeatureSubRegion<Feature> {
        unsafe {
            &mut*(self as *mut Chunk2dCacheLine as *mut FeatureSubRegion<Feature>)
        }
    }
}

impl Chunk2dCacheLine {
    #[inline(always)]
    fn as_simd<const LANES: usize>(&self) -> &[Simd<u8, LANES>]
    where
        simd::LaneCount<LANES>: simd::SupportedLaneCount,
    {
/*        let this: &[u8; 64] = unsafe {
            &*(self as *const Self as *const [u8; 64])
        };*/
        let (pre, simd, post) = self.0.as_flattened().as_simd();
        assert_eq!(align_of::<Self>(), 64);
        assert_eq!(64, LANES * simd.len());
        if (pre.len() != 0 || post.len() != 0) {
            unsafe { unreachable_unchecked() }
        }
        simd
    }

    #[inline(always)]
    fn as_simd64(&self) -> &SimdM<u8, 64> {
        let lanes = self.as_simd::<64>();
        assert_eq!(lanes.len(), 1);
        &lanes[0]
    }

    #[inline(always)]
    fn as_simd_mut<const LANES: usize>(&mut self) -> &mut [Simd<u8, LANES>]
    where
        simd::LaneCount<LANES>: simd::SupportedLaneCount,
    {
        /*        let this: &[u8; 64] = unsafe {
                    &*(self as *const Self as *const [u8; 64])
                };*/
        let (pre, simd, post) = self.0.as_flattened_mut().as_simd_mut();
        assert_eq!(pre.len(), 0);
        assert_eq!(post.len(), 0);
        simd
    }


    #[inline(always)]
    fn as_simd64_mut(&mut self) -> &mut Simd<u8, 64> {
        let lanes = self.as_simd_mut::<64>();
        assert_eq!(lanes.len(), 1);
        &mut lanes[0]
    }

}

impl IndexMut<Coord> for Chunk2dCacheLine {
    #[inline(always)]
    fn index_mut(&mut self, index: Coord) -> &mut Self::Output {
        index.get_single_flag_mut(self)
    }
}
impl IndexMut<u8> for Chunk2dCacheLine {
    #[inline(always)]
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self[index as usize]
    }
}
impl IndexMut<usize> for Chunk2dCacheLine {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}


impl Index<Coord> for Chunk2dCacheLine {
    type Output = u8;

    #[inline(always)]
    fn index(&self, index: Coord) -> &Self::Output {
        &index.get_single_flag_ref(self)
    }
}

impl Index<u8> for Chunk2dCacheLine {
    type Output = [u8;8];

    #[inline(always)]
    fn index(&self, index: u8) -> &Self::Output {
        &self[index as usize]
    }
}

impl Index<usize> for Chunk2dCacheLine {
    type Output = [u8;8];

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}


#[repr(transparent)]
struct FeatureSubRegion<Feature: TileWorldFeatureAccessorRaw>(Chunk2dCacheLine, PhantomData<Feature>);

impl<Feature: TileWorldFeatureAccessorRaw> FeatureSubRegion<Feature> {

    /// if mask is 0000###0 afterward each
    /// value is   00000###
    /// rest of date is replaced with 0 and value is low aligned
    #[inline(always)]
    fn simd_mask_shift(simd: &mut Simd<u8, 64>) {
        //all these ifs use CONST, so only the active branch actually is compiled
        if Feature::BITS == 8 {return}
        if Feature::OFFSET == 0 {
            *simd &= u8x64::splat(Feature::VAL_MASK);
        } else {
            if Feature::VAL_MASK_LEFT_SHIFT > 0 {
                *simd <<= Feature::VAL_MASK_LEFT_SHIFT;
            }
            *simd >>= Feature::VAL_MASK_RIGHT_SHIFT;
        }
    }

    /// this masks the value but does not move them
    ///if low aligned values are required simd_mask_shift is more efficient
    #[inline(always)]
    fn simd_mask(simd: &mut Simd<u8, 64>) {
        if Feature::BITS < 8 {
            *simd &= u8x64::splat(Feature::VAL_MASK);
        }
    }

    #[inline(always)]
    fn simd_low_align(simd: &mut Simd<u8, 64>) {
        if Feature::OFFSET > 0 {
            *simd >>= Feature::OFFSET;
        }
    }


    #[inline(always)]
    fn simd_unshift(simd: &mut Simd<u8, 64>) {
        if Feature::OFFSET > 0 {
            *simd <<= Feature::OFFSET;
        }
    }

    #[inline(always)]
    fn simd_prep_val<const MASK: bool, const SHIFT: bool>(simd: &mut Simd<u8, 64>) {
        match (MASK, SHIFT) {
            (true, true) => Self::simd_mask_shift(simd),
            (true, false) => Self::simd_mask(simd),
            (false, true) => Self::simd_low_align(simd),
            _ => {}
        }
    }


    #[inline(always)]
    fn simd_unprep_val<const MASK: bool, const SHIFT: bool, const CLEAN: bool>(simd: &mut Simd<u8, 64>) {
        //if BITS == 8 simd_unshift() and simd_mask() are empty
        if SHIFT {
            Self::simd_unshift(simd);
        }
        if !CLEAN {
            Self::simd_mask(simd);
        }
    }

    /// MASK will replace all other stored data with zero
    /// SHIFT will low align data
    /// CLEAN an unclean function will pollute storage that would be MASKED
    #[inline(always)]
    fn apply_selected<const MASK: bool, const SHIFT: bool, const CLEAN: bool>
    (&mut self, func_simd: fn(&mut Simd<u8, 64>) -> mask8x64)
    {
        let mut simd = *self.0.as_simd64_mut();
        Self::simd_prep_val::<MASK, SHIFT>(&mut simd);

        let mask = func_simd(&mut simd);

        Self::simd_unprep_val::<MASK, SHIFT, CLEAN>(&mut simd);
        if (Feature::BITS < 8) {
            simd |= *self.0.as_simd64_mut() & u8x64::splat(Feature::ERASE_MASK);
        }
        simd.store_select(self.0.0.as_flattened_mut(), mask);
    }

    /// MASK will replace all other stored data with zero
    /// SHIFT will low align data
    /// CLEAN an unclean function will pollute storage that would be MASKED
    #[inline(always)]
    fn apply<const MASK: bool, const SHIFT: bool, const CLEAN: bool>
    (&mut self, func_simd: fn(&mut Simd<u8, 64>))
    {
        let mut simd = *self.0.as_simd64_mut();
        Self::simd_prep_val::<MASK, SHIFT>(&mut simd);

        func_simd(&mut simd);

        Self::simd_unprep_val::<MASK, SHIFT, CLEAN>(&mut simd);
        let this = self.0.as_simd64_mut();
        if (Feature::BITS < 8) {
            *this &= u8x64::splat(Feature::ERASE_MASK);
            *this |= simd;
        } else {
            *this = simd;
        }
    }

    /// MASK will replace all other stored data with zero
    /// SHIFT will low align data
    #[inline(always)]
    fn inspect<const MASK: bool, const SHIFT: bool, Result>
    (&self, func_simd: fn(&SimdM<u8, 64>) -> Result) -> Result
    {
        let mut simd = *self.0.as_simd64();
        Self::simd_prep_val::<MASK, SHIFT>(&mut simd);

        func_simd(&simd)
    }

    /// MASK will replace all other stored data with zero
    /// SHIFT will low align data
    #[inline(always)]
    fn inspect_typed<const MASK: bool, const SHIFT: bool, Result, T>
    (&self, func_simd: fn(TypedSimd<T, &SimdM<u8, 64>>) -> Result) -> Result
    where Feature : TileWorldFeature<Layout, T>,
          Layout: TileWorldLayoutDesc,
          T: FeatureType<SimdRepl = u8>
    {
        let mut simd = *self.0.as_simd64(); //if MASK & SHIFT both false, we dont need this copy
        Self::simd_prep_val::<MASK, SHIFT>(&mut simd);


        func_simd(TypedSimd::new(&simd))
    }
}

type TileWorldData<Layout: TileWorldLayoutDesc> = [Chunk2dPageSize;Layout::LAYERS as usize];

struct TileWorld<Layout: TileWorldLayoutDesc> where [(); Layout::LAYERS as usize]:{
    data: Box<[TileWorldData<Layout>]>,
    height: u16, width: u16,
    region_height: u16, region_width: u16,
}


trait TileWorldLayoutDesc{
    const LAYERS: u16;
}

trait TileWorldFeatureAccessorRaw {
    const LAYER: u16;                                                           //e.g. 00011100
    const OFFSET: u8;                                                           // =2        __
    const BITS: u8;                                                             // =3     ___
    const VAL_MASK_LEFT_SHIFT: u8 = 8 - Self::OFFSET - Self::BITS;              // =3  ___
    const VAL_MASK_RIGHT_SHIFT: u8 = 8 - Self::VAL_MASK_LEFT_SHIFT;             // =5     _____
    const VAL_MASK: u8 = (((1u16 << Self::BITS) - 1) as u8) << Self::OFFSET;    //     00011100
    const ERASE_MASK: u8 = !Self::VAL_MASK;                                     //     11100011
    const LAYER_MAJOR: u16 = Self::LAYER >> 2;
    const LAYER_MINOR: u16 = Self::LAYER & 0b0011;
    
}

trait TileWorldFeatureAccessor<Layout: TileWorldLayoutDesc> : TileWorldFeatureAccessorRaw {
    
}

trait TileWorldFeature<Layout: TileWorldLayoutDesc, T: Copy> : TileWorldFeatureAccessor<Layout> {

    fn map(bits: u8) -> T;
    fn unmap(val: T) -> u8;

    #[inline(always)]
    fn set(bits: &mut u8, val: T) {
        *bits = (*bits & Self::ERASE_MASK) | (Self::unmap(val))
    }


}


impl<Layout: TileWorldLayoutDesc> TileWorld<Layout> where [(); Layout::LAYERS as usize]: {

    fn new(width: u16, height: u16) -> Self {
        let region_height = height >> 3 + (height & 7 != 0) as u16;
        let region_width = width >> 3 + (width & 7 != 0) as u16;
        let data = Box::new_zeroed_slice((region_height * region_width) as usize);
        Self {
            data: unsafe{data.assume_init()},
            height, width,
            region_height, region_width
        }
    }


    #[inline(always)]
    const fn get_region_arr(&self, coords: Coord) -> &TileWorldData<Layout>  {
        &self.data[coords.region_index(self.region_width)]
    }

    #[inline(always)]
    const fn get_region_arr_mut(&mut self, coords: Coord) -> &mut TileWorldData<Layout>  {
        &mut self.data[coords.region_index(self.region_width)]
    }


    #[inline(always)]
    const fn get_region<Accessor>(&self, coords: Coord) -> &Chunk2dCacheLine
    where Accessor: TileWorldFeatureAccessor<Layout>
    {
        &self.get_region_arr(coords)[Accessor::LAYER_MAJOR as usize][coords][Accessor::LAYER_MINOR as usize]
    }


    #[inline(always)]
    const fn get_region_mut<Accessor>(&mut self, coords: Coord) -> &mut Chunk2dCacheLine
    where Accessor: TileWorldFeatureAccessor<Layout>
    {
        &mut self.get_region_arr_mut(coords)[Accessor::LAYER_MAJOR as usize][coords][Accessor::LAYER_MINOR as usize]
    }


    #[inline(always)]
    fn get<Accessor, T>(&self, coords: Coord) -> T
    where Accessor: TileWorldFeature<Layout, T>, T: Copy
    {
        Accessor::map(coords.get_single_flag(self.get_region::<Accessor>(coords)))
    }

    #[inline(always)]
    fn set<Accessor, T>(&mut self, coords: Coord, val: T)
    where Accessor: TileWorldFeature<Layout, T>, T: Copy
    {
        Accessor::set(coords.get_single_flag_mut(self.get_region_mut::<Accessor>(coords)), val);
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct Coord {
    region_left: u8,
    region_top: u8,
    region_subindex: u8,
    chunk_subindex: u8,
}

impl Coord {
    #[inline(always)]
    const fn new(left: usize, top: usize) -> Self {
        Self {
            region_left: (left >> 5) as u8,
            region_top: (top >> 5) as u8,
            region_subindex: ((left as u8 & 0b0001_1000) >> 1) | ((top as u8 & 0b0001_1000) << 1),
            chunk_subindex: (left as u8 & 0b0000_0111) | ((top as u8 & 0b0000_0111) << 3),
        }
    }

    #[inline(always)]
    const fn chunk_left(self) -> u8 {
        self.chunk_subindex & 0b111
    }

    #[inline(always)]
    const fn chunk_top(self) -> u8 {
        self.chunk_subindex >> 3
    }

    #[inline(always)]
    const fn subregion_left(self) -> u8 {
        self.chunk_subindex & 0b11
    }

    #[inline(always)]
    const fn subregion_top(self) -> u8 {
        self.chunk_subindex >> 2
    }

    #[inline(always)]
    const fn left(self) -> usize {
        ((self.region_left as usize) << 3) | self.chunk_left() as usize
    }

    #[inline(always)]
    const fn top(self) -> usize {
        ((self.region_left as usize) << 3) | self.chunk_top() as usize
    }

    #[inline(always)]
    const fn region_index(self, region_width: u16) -> usize {
        self.region_left as usize + self.region_top as usize * region_width as usize
    }


    #[inline(always)]
    const fn get_single_flag(self, data: &Chunk2dCacheLine) -> u8 {
        //if we use the Index impl it's no longer const-able....
        data.0[self.chunk_left() as usize][self.chunk_top() as usize]
    }
    #[inline(always)]
    const fn get_single_flag_ref(self, data: &Chunk2dCacheLine) -> &u8 {
        //if we use the Index impl it's no longer const-able....
        &data.0[self.chunk_left() as usize][self.chunk_top() as usize]
    }

    #[inline(always)]
    const fn get_single_flag_mut(self, data: &mut Chunk2dCacheLine) -> &mut u8 {
        &mut data.0[self.chunk_left() as usize][self.chunk_top() as usize]
    }

    #[inline(always)]
    const fn set_single_flag(self, data: &mut Chunk2dCacheLine, val: u8) {
        data.0[self.chunk_left() as usize][self.chunk_top() as usize] = val;
    }
    
}
#[repr(u8)]
enum Test{
    first = 1,
    second = 2,
}


#[derive(Copy, Clone, PartialEq)]
enum DataValidity {
    Unchecked,
    Valid,
    Invalid
}
#[derive(Debug, Default)]
struct DataInvalidError();

impl Display for DataInvalidError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataInvalidError")
    }
}

impl Error for DataInvalidError {}

impl Into<DataValidity> for DataInvalidError {
    fn into(self) -> DataValidity {
        Invalid
    }
}

mod same_size {
    use std::mem;
    use std::mem::ManuallyDrop;

    trait SameSizeAsSealed<T2> {}
    trait SameAsSealed<T2> {}
    pub trait SameAs<T> : SameAsSealed<T>{
        fn into_same(self) -> T;
    }
    impl<T> SameAs<T> for T{
        fn into_same(self) -> T {
            self
        }
    }
    impl<T, T2> SameAsSealed<T2> for T
    where T: SameAs<T2>{}

    pub trait SameSizeAs<T>: SameSizeAsSealed<T> + Sized {
        #[inline(always)]
        fn yes_same_array<T2>(arr: [T2; size_of::<Self>()]) -> [T2; size_of::<T>()] {
            union UnionTransmute<From, To> {
                from: ManuallyDrop<From>,
                to: ManuallyDrop<To>,
            }

            let union = UnionTransmute {
                from: ManuallyDrop::new(arr),
            };

            ManuallyDrop::into_inner( unsafe {union.to})
        }
    }
    impl<T, T2> SameSizeAs<T2> for T
    where [u8; size_of::<T>()]: SameAs<[u8; size_of::<T2>()]> {}
    impl<T, T2> SameSizeAsSealed<T2> for T
    where T: SameSizeAs<T2>{}

    pub trait FlattenConstSizeArr<T, const D1: usize,const D2: usize> where Self: SameAs<[[T;D1];D2]> {
        fn flatten_ref(&self) -> &[T;(D1*D2)];
        fn flatten_mut(&mut self) -> &mut[T;(D1*D2)];
    }

    //todo check if we must enforce alignment?
    impl<T, const D1: usize,const D2: usize> FlattenConstSizeArr<T, D1, D2> for [[T;D1];D2]  {
        fn flatten_ref(&self) -> &[T; (D1*D2)] {
            unsafe {
                &*(self as *const _ as *const [T; (D1*D2)])
            }
        }

        fn flatten_mut(&mut self) -> &mut [T; (D1*D2)] {
            unsafe {
                &mut *(self as *mut _ as *mut [T; (D1*D2)])
            }
        }
    }

}

trait FeatureTypeValidator<SimdRepl: SimdElement + SameSizeAs<Self> + SimdHelper::SignedPair2, Validator: ValidatorNameZST>: Sized {
    fn simd_check<const LANES:usize>(data: &SimdM<SimdRepl, LANES>) -> bool
    where LaneCount<LANES>: SupportedLaneCount,
          SimdM<SimdRepl, LANES>: SimdMathable<SimdRepl, LANES>,
          MaskInner<SimdRepl, LANES>: MaskMathable<SimdRepl, LANES>;
}

trait ValidatorNameZST : SameSizeAs<()>{}

trait FeatureType: Copy + SameSizeAs<Self::SimdRepl> + FeatureTypeValidator<Self::SimdRepl, Self::Validator> {
    type SimdRepl: SimdElement + SameSizeAs<Self> + Eq + SignedPair2;
    type Validator: ValidatorNameZST;

}

unsafe trait NoFeatureTypeValidator: FeatureType{}
struct NoFeatureTypeValidatorName{}
impl ValidatorNameZST for NoFeatureTypeValidatorName{}
impl<T> FeatureTypeValidator<T::SimdRepl, NoFeatureTypeValidatorName> for T
where
    T: NoFeatureTypeValidator + FeatureType<Validator=NoFeatureTypeValidatorName>
{
    fn simd_check<const LANES: usize>(data: &SimdM<T::SimdRepl, LANES>) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount
    {
        true
    }
}


unsafe trait RangeFeatureTypeValidator: FeatureType{
    const MIN: Self::SimdRepl;
    const MAX: Self::SimdRepl;
}
struct RangeFeatureTypeValidatorName{}
impl ValidatorNameZST for RangeFeatureTypeValidatorName{}
impl<T> FeatureTypeValidator<T::SimdRepl, RangeFeatureTypeValidatorName> for T
where
    T: RangeFeatureTypeValidator + FeatureType<Validator=RangeFeatureTypeValidatorName>
{
    fn simd_check<const LANES:usize>(data: &SimdM<T::SimdRepl, LANES>) -> bool
    where LaneCount<LANES>: SupportedLaneCount,
          SimdM<T::SimdRepl, LANES>: SimdMathable<T::SimdRepl, LANES>,
          MaskInner<T::SimdRepl, LANES>: MaskMathable<T::SimdRepl, LANES>,
    {
        let min = SimdM::<T::SimdRepl, LANES>::splat(T::MIN);
        let max = SimdM::<T::SimdRepl, LANES>::splat(T::MAX);

        let ge_min = (*data).simd_ge(min);
        let le_max = (*data).simd_le(max);
        
        let x: MaskM<T::SimdRepl, LANES> = (ge_min & le_max).into_same();
        
        
        //x.all()
        true
    }
}


impl FeatureType for i8 {
    type SimdRepl = u8;
    type Validator = NoFeatureTypeValidatorName;
}
unsafe impl NoFeatureTypeValidator for i8 {
}


struct TypedSimd<TTyped, Data>
where
    TTyped: FeatureType,
    Data: Borrow<SimdM<TTyped::SimdRepl, 64>>,
    SimdM<TTyped::SimdRepl, 64>: SimdMathable<TTyped::SimdRepl, 64>,
    MaskInner<TTyped::SimdRepl, 64>: MaskMathable<TTyped::SimdRepl, 64>,
{
    simd: Data,
    checked: Cell<DataValidity>,
    _phantom_data: PhantomData<[[[TTyped;8];8]]>
}

mod SimdHelper {
    use std::ops::{BitAnd, BitOr};
    use std::simd::cmp::SimdPartialOrd;
    use std::simd::{LaneCount, Mask, MaskElement, Simd, SimdElement, SupportedLaneCount};
    use std::simd::prelude::SimdPartialEq;
    use crate::same_size::SameAs;

    trait Sealed{}

    pub type MaskInner<T, const N: usize>
    where Simd<T, N>: SimdMathable<T, N>,
          MaskInner<T, N>: MaskMathable<T, N> + BitAnd<Output=MaskInner<T, N>> + BitOr<Output=MaskInner<T, N>>,
          T: SignedPair2,

    = <Simd<T, N> as SimdPartialEq>::Mask;

    pub type MaskM<T, const N: usize>
    where Simd<T, N>: SimdMathable<T, N>,
          MaskInner<T, N>: MaskMathable<T, N>
          + BitAnd<Output=MaskM<T, N>>
          + BitOr<Output=MaskM<T, N>>,
          Mask<T::Signed, N>: SameAs<<Simd<T, N> as SimdPartialEq>::Mask>,
          T: SignedPair2,

    = Mask<T::Signed, N>;

    pub type SimdM<T, const N: usize>
    where
        SimdM<T, N>: SimdMathable<T, N>,
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + SignedPair2,
        MaskM<T, N>: MaskMathable<T, N>
    = Simd<T, N>;


    trait Signed: SignedPair<Self, <Self as Signed>::Unsigned> + MaskElement{
        type Unsigned: Unsigned<Signed=Self>;
    }

    trait SignedPair<TSigned: Signed + MaskElement, TUnSigned: Unsigned>: Sized{}
    trait Unsigned : SignedPair<Self::Signed, Self>{
        type Signed: Signed<Unsigned=Self>;
    }

    pub trait SignedPair2{
        type Signed: Signed<Unsigned=Self::Unsigned>;
        type Unsigned: Unsigned<Signed=Self::Signed>;
    }

    macro_rules! ImplSignedPair {
    ($signed:ty, $unsigned:ty) => {
            impl Unsigned for $unsigned {
                type Signed = $signed;
            }
            impl Signed for $signed {
                type Unsigned = $unsigned;
            }
            impl SignedPair2 for $unsigned {
                type Signed = $signed;
                type Unsigned = $unsigned;
            }
            impl SignedPair2 for $signed {
                type Signed = $signed;
                type Unsigned = $unsigned;
            }
            impl SignedPair<$signed, $unsigned> for $signed{}
            impl SignedPair<$signed, $unsigned> for $unsigned{}
        }
    }
    ImplSignedPair!(i8, u8);
    ImplSignedPair!(i16, u16);
    ImplSignedPair!(i32, u32);
    ImplSignedPair!(i64, u64);
    ImplSignedPair!(isize, usize);




    pub trait MaskMathable<T, const N: usize> : SimdPartialOrd + Sized/* + SameAs<MaskM<T, N>>*/
    where
        Self: BitAnd<Output=Self> + BitOr<Output=Self> + SameAs<MaskM<T, N>> /*+ SameAs<MaskInner<T, N>>*/ + SimdPartialEq,
        T: SimdElement + SignedPair2,
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: SimdPartialEq
    {}
    
    impl<T, const N: usize> MaskMathable<T, N> for MaskM<T, N>
    where 
        Self: SimdPartialOrd + SimdPartialEq + Sized,
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement + SignedPair2,
        Simd<T, N>: SimdPartialEq,
        Self: BitAnd<Output=Self> + BitOr<Output=Self>
    {}

    
    pub trait SimdMathable<T, const N: usize> : SimdPartialOrd + Sealed
    where
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement,
        Self::Mask: BitAnd<Output=Self::Mask> + BitOr<Output=Self::Mask> /*+ BitAndAssign<Self::Mask> + BitOrAssign<Self::Mask>*/
    {}

    impl<T, const N: usize> SimdMathable<T, N> for SimdM<T, N>
    where
        Self: SimdPartialOrd,
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement,
        Self::Mask: BitAnd<Output=Self::Mask> + BitOr<Output=Self::Mask> /*+ BitAndAssign<Self::Mask> + BitOrAssign<Self::Mask>*/
    {}

    impl<T, const N: usize> Sealed for SimdM<T, N> where
        Self: SimdPartialOrd,
        LaneCount<N>: SupportedLaneCount,
        T: SimdElement,
        <Self as SimdPartialEq>::Mask: BitAnd<Output=<Self as SimdPartialEq>::Mask> + BitOr<Output=<Self as SimdPartialEq>::Mask> /*+ BitAndAssign<Self::Mask> + BitOrAssign<Self::Mask>*/
    {}
    
}


impl<TTyped, Data> TypedSimd<TTyped, Data>
where
    TTyped: FeatureType,
    Data: BorrowMut<SimdM<TTyped::SimdRepl, 64>>,
    SimdM<TTyped::SimdRepl, 64>: SimdMathable<TTyped::SimdRepl, 64>,
    MaskInner<TTyped::SimdRepl, 64>: MaskMathable<TTyped::SimdRepl, 64>,
    LaneCount<64>: SupportedLaneCount
{
    pub fn simd_mut(&mut self) -> &mut Simd<TTyped::SimdRepl, 64>  {
        self.checked.set(Unchecked);
        self.simd.borrow_mut()
    }


    pub fn typed_mut(&mut self) -> Result<&mut [[TTyped;8];8],DataInvalidError>  {
        self.check()?;
        Ok(unsafe {
            &mut*(self.simd.borrow_mut() as *mut _ as *mut [[TTyped;8];8])
        })
    }
}

impl<TTyped, Data> TypedSimd<TTyped, Data>
where
    TTyped: FeatureType,
    Data: Borrow<SimdM<TTyped::SimdRepl, 64>>,
    SimdM<TTyped::SimdRepl, 64>: SimdMathable<TTyped::SimdRepl, 64>,
    MaskInner<TTyped::SimdRepl, 64>: MaskMathable<TTyped::SimdRepl, 64>,
{
    pub fn new(data: Data) -> Self {
        Self {
            simd: data,
            checked: Cell::new(Unchecked),
            _phantom_data: PhantomData
        }
    }

    pub fn simd(&self) -> &SimdM<TTyped::SimdRepl, 64> {
        self.simd.borrow()
    }


    /// checks if the data is valid for TTyped is it may have gotten invalidated (e.g. by simd_mut)
    pub fn check(&self) -> Result<(),DataInvalidError>  {
        if self.checked.get() == Unchecked {
            self.checked.set(match TTyped::simd_check(self.simd()) {
                true => Valid,
                false => Invalid
            });
        }
        match self.checked.get() {
            Invalid => Err(DataInvalidError()),
            _ => Ok(())
        }
    }

    pub fn typed(&self) -> Result<&[[TTyped;8];8],DataInvalidError> {
        self.check()?;
        Ok(unsafe {
            &*(self.simd.borrow() as *const _ as *const [[TTyped;8];8])
        })
    }
}


#[derive(Copy, Clone)]
#[repr(u8)]
enum TestFeature{
    Grassland = 1,
    Forest = 2,

}

impl FeatureType for TestFeature {
    type SimdRepl = u8;
    type Validator = NoFeatureTypeValidatorName;
}

unsafe impl NoFeatureTypeValidator for TestFeature{}
struct TestFeatureLayout();

fn main() {
    println!("{}", size_of::<NotZero>());
    println!("{}", size_of::<Option<NotZero>>());
    let x = NotZero::two(2135523);
    println!("{}", x.val());

    let x = [3u8, 0];
    let mut simd = x.as_simd::<2>().1[0];
    simd >>= 1;
    println!("{:?}", simd.as_array());


    println!("Hello, world!");
}
