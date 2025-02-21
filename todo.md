it seems i made a small mistake and assumed that the max length of a SIMD is 512 bits
i think my assumption is correct but RUST does support any number type as 64 lanes. 
thus it seems to be that i do not need to care about dealing with arrays of SIMD but can always fallback on 64 lanes and let the compiler figure it out for me

this actually is good because it means i will never have to coerce SIMD types which i currently do at one place
but it also requires me to rewrite Typed_Simd which i dont have time for now