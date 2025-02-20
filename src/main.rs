#![feature(new_zeroed_alloc)]
#![feature(generic_const_exprs)]
#![feature(portable_simd)]
#![feature(trivial_bounds)]
#![feature(transparent_unions)]
#![feature(repr_simd)]

use core::simd;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut};
use std::simd::{mask8x64, u8x64, Simd, SimdElement};
use not_zero::NotZero;
mod not_zero;

#[repr(align(64))]
struct SubRegionAligned([[u8;8];8]);

impl<'a, Feature: TileWorldFeatureAccessorRaw> Into<&'a FeatureSubRegion<Feature>> for &'a SubRegionAligned {
    fn into(self) -> &'a FeatureSubRegion<Feature> {
        unsafe {
            &*(self as *const SubRegionAligned as *const FeatureSubRegion<Feature>)
        }
    }
}

impl<'a, Feature: TileWorldFeatureAccessorRaw> Into<&'a mut FeatureSubRegion<Feature>> for &'a mut SubRegionAligned {
    fn into(self) -> &'a mut FeatureSubRegion<Feature> {
        unsafe {
            &mut*(self as *mut SubRegionAligned as *mut FeatureSubRegion<Feature>)
        }
    }
}

impl SubRegionAligned {
    #[inline(always)]
    fn as_simd<const LANES: usize>(&self) -> &[Simd<u8, LANES>]
    where
        simd::LaneCount<LANES>: simd::SupportedLaneCount,
    {
/*        let this: &[u8; 64] = unsafe {
            &*(self as *const Self as *const [u8; 64])
        };*/
        let (pre, simd, post) = self.0.as_flattened().as_simd();
        assert_eq!(pre.len(), 0);
        assert_eq!(post.len(), 0);
        simd
    }

    #[inline(always)]
    fn as_simd64(&self) -> &Simd<u8, 64> {
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

impl IndexMut<Coords> for SubRegionAligned {
    #[inline(always)]
    fn index_mut(&mut self, index: Coords) -> &mut Self::Output {
        index.get_single_flag_mut(self)
    }
}
impl IndexMut<u8> for SubRegionAligned {
    #[inline(always)]
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self[index as usize]
    }
}
impl IndexMut<usize> for SubRegionAligned {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}


impl Index<Coords> for SubRegionAligned {
    type Output = u8;

    #[inline(always)]
    fn index(&self, index: Coords) -> &Self::Output {
        &index.get_single_flag_ref(self)
    }
}

impl Index<u8> for SubRegionAligned {
    type Output = [u8;8];

    #[inline(always)]
    fn index(&self, index: u8) -> &Self::Output {
        &self[index as usize]
    }
}

impl Index<usize> for SubRegionAligned {
    type Output = [u8;8];

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}


#[repr(transparent)]
struct FeatureSubRegion<Feature: TileWorldFeatureAccessorRaw>(SubRegionAligned, PhantomData<Feature>);

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
    (&self, func_simd: fn(&mut Simd<u8, 64>) -> Result) -> Result
    {
        let mut simd = *self.0.as_simd64();
        Self::simd_prep_val::<MASK, SHIFT>(&mut simd);

        func_simd(&mut simd)
    }

    /// MASK will replace all other stored data with zero
    /// SHIFT will low align data
    #[inline(always)]
    const fn inspect_typed<const MASK: bool, const SHIFT: bool, Result, T>
    (&self, func_simd: fn(&mut Simd<TypedSimd<T, u8>, 64>) -> Result) -> Result
    where Feature : TileWorldFeature<Layout, T>,
          Layout: TileWorldLayoutDesc,
          T: Copy
    {
        let mut simd = *self.0.as_simd64();
        Self::simd_prep_val::<MASK, SHIFT>(&mut simd);


        func_simd(&mut simd)
    }
}

type TileWorldData<Layout: TileWorldLayoutDesc> = [SubRegionAligned;Layout::LAYERS as usize];

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
    const fn get_region_arr(&self, coords: Coords) -> &TileWorldData<Layout>  {
        &self.data[coords.region_index(self.region_width)]
    }

    #[inline(always)]
    const fn get_region_arr_mut(&mut self, coords: Coords) -> &mut TileWorldData<Layout>  {
        &mut self.data[coords.region_index(self.region_width)]
    }


    #[inline(always)]
    const fn get_region<Accessor>(&self, coords: Coords) -> &SubRegionAligned
    where Accessor: TileWorldFeatureAccessor<Layout>
    {
        &self.get_region_arr(coords)[Accessor::LAYER as usize]
    }


    #[inline(always)]
    const fn get_region_mut<Accessor>(&mut self, coords: Coords) -> &mut SubRegionAligned
    where Accessor: TileWorldFeatureAccessor<Layout>
    {
        &mut self.get_region_arr_mut(coords)[Accessor::LAYER as usize]
    }


    #[inline(always)]
    fn get<Accessor, T>(&self, coords: Coords) -> T
    where Accessor: TileWorldFeature<Layout, T>, T: Copy
    {
        Accessor::map(coords.get_single_flag(self.get_region::<Accessor>(coords)))
    }

    #[inline(always)]
    fn set<Accessor, T>(&mut self, coords: Coords, val: T)
    where Accessor: TileWorldFeature<Layout, T>, T: Copy
    {
        Accessor::set(coords.get_single_flag_mut(self.get_region_mut::<Accessor>(coords)), val);
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct Coords{
    region_left: u16,
    region_top: u16,
    region_subindex: u8,
}

impl Coords {
    #[inline(always)]
    const fn new(left: usize, top: usize) -> Self {
        Self {
            region_left: (left >> 3) as u16,
            region_top: (top >> 3) as u16,
            region_subindex: (left as u8 & 7) | ((top as u8 & 7) << 3),
        }
    }

    #[inline(always)]
    const fn subindex_left(self) -> u8 {
        self.region_subindex & 7
    }

    #[inline(always)]
    const fn subindex_top(self) -> u8 {
        self.region_subindex & 7
    }

    #[inline(always)]
    const fn left(self) -> usize {
        ((self.region_left as usize) << 3) | self.subindex_left() as usize
    }

    #[inline(always)]
    const fn top(self) -> usize {
        ((self.region_left as usize) << 3) | self.subindex_top() as usize
    }

    #[inline(always)]
    const fn region_index(self, region_width: u16) -> usize {
        self.region_left as usize + self.region_top as usize * region_width as usize
    }


    #[inline(always)]
    const fn get_single_flag(self, data: &SubRegionAligned) -> u8 {
        //if we use the Index impl it's no longer const-able....
        data.0[self.subindex_left() as usize][self.subindex_top() as usize]
    }
    #[inline(always)]
    const fn get_single_flag_ref(self, data: &SubRegionAligned) -> &u8 {
        //if we use the Index impl it's no longer const-able....
        &data.0[self.subindex_left() as usize][self.subindex_top() as usize]
    }

    #[inline(always)]
    const fn get_single_flag_mut(self, data: &mut SubRegionAligned) -> &mut u8 {
        &mut data.0[self.subindex_left() as usize][self.subindex_top() as usize]
    }

    #[inline(always)]
    const fn set_single_flag(self, data: &mut SubRegionAligned, val: u8) {
        data.0[self.subindex_left() as usize][self.subindex_top() as usize] = val;
    }
    
}
#[repr(u8)]
enum Test{
    first = 1,
    second = 2,
}

#[repr(transparent)]
struct TypedSimd<TTyped:Copy, TSimdElement: SimdElement, const LANES:usize> {
    simd: [Simd<TSimdElement, 64>; size_of::<TTyped>()],
    _phantom_data: PhantomData<[[[TTyped;8];8]]>
}

impl<TTyped: Copy, TSimdElement: SimdElement, const LANES: usize> TypedSimd<TTyped, TSimdElement, LANES> {
    pub fn new(data: [Simd<TSimdElement, 64>; size_of::<TTyped>()]) -> Self {
        todo!()
        //todo
    }
}

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
