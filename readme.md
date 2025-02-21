the goal of the project is to use the rust type system to allow for zero-cost abstractions on bitfield storage in a tileworld that is typesafe 

the zero-cost is realised by having all features be defined solely using empty, compile-time only types (ZST) and the use of generic, inlineable fn that are populated by the const in the ZST

most operations should be realised by SIMD operations. the unit of SIMD is one 512 bit lane representing 8x8 tiles with u8 storage capacity each. 

The usage of this 8bit storage may utilise bitfields, may consume the whole u8 or may combine multiple u8 to larger data types. any bit count is possible. This is completely defined by the ZST defining a tile feature. It is possible to do any SIMD operation on these values, but housekeeping is completely performed by this library using the compile time specialisation selection based on the ZST's associated consts. However, any housekeeping can be opted out if such an operation is not need for the high performance SIMD operation. 


there is one giant caveat to all this: This project is not necessary safe if code consuming this library performs SIMD operations resulting in values that are not valid values of the rust type: when converting from bit representation into rust types no sanity/safety checks are performed. 

this does mean that any simd function modifying state should be marked as unsafe, but i am unhappy with how much this would pollute code with unsafe assertions to show that what one does is in fact not unsafe. also as math can be harder to proof correct this assertion may anyway often be wrong...
I have actually introduced a good way of dealing with this for all vectorized accesses
i currently just have no way of enforcing this for unvectorized accesses without having to check every time. maybe thats what i just have to do. vectorized accesses should not be used with any frequency anyway