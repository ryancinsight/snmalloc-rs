#![no_std]
//! `snmalloc-rs` provides a wrapper for [`microsoft/snmalloc`](https://github.com/microsoft/snmalloc) to make it usable as a global allocator for rust.
//! snmalloc is a research allocator. Its key design features are:
//! - Memory that is freed by the same thread that allocated it does not require any synchronising operations.
//! - Freeing memory in a different thread to initially allocated it, does not take any locks and instead uses a novel message passing scheme to return the memory to the original allocator, where it is recycled.
//! - The allocator uses large ranges of pages to reduce the amount of meta-data required.
//!
//! The benchmark is available at the [paper](https://github.com/microsoft/snmalloc/blob/master/snmalloc.pdf) of `snmalloc`
//! There are three features defined in this crate:
//! - `debug`: Enable the `Debug` mode in `snmalloc`.
//! - `1mib`: Use the `1mib` chunk configuration.
//! - `cache-friendly`: Make the allocator more cache friendly (setting `CACHE_FRIENDLY_OFFSET` to `64` in building the library).
//!
//! The whole library supports `no_std`.
//!
//! To use `snmalloc-rs` add it as a dependency:
//! ```toml
//! # Cargo.toml
//! [dependencies]
//! snmalloc-rs = "0.1.0"
//! ```
//!
//! To set `SnMalloc` as the global allocator add this to your project:
//! ```rust
//! #[global_allocator]
//! static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;
//! ```
extern crate snmalloc_sys as ffi;

use core::{
    alloc::{GlobalAlloc, Layout},
    ptr::{self,NonNull},
};

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct SnMalloc;

unsafe impl Send for SnMalloc {}
unsafe impl Sync for SnMalloc {}

#[repr(align(16))]
struct ZstSentinel;

static ZST_SENTINEL: ZstSentinel = ZstSentinel;

impl SnMalloc {
    #[inline(always)]
    pub const fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn handle_zst(&self) -> NonNull<u8> {
        unsafe { NonNull::new_unchecked(&ZST_SENTINEL as *const _ as *mut u8) }
    }

    /// Allocates memory with the given layout.
    #[inline(always)]
    pub fn safe_alloc(&self, layout: Layout) -> Option<NonNull<u8>> {
        match layout.size() {
            0 => Some(self.handle_zst()),
            size => NonNull::new(unsafe { ffi::sn_rust_alloc(layout.align(), size) }.cast()),
        }
    }

    /// Allocates zero-initialized memory with the given layout.
    #[inline(always)]
    pub fn safe_alloc_zeroed(&self, layout: Layout) -> Option<NonNull<u8>> {
        match layout.size() {
            0 => Some(self.handle_zst()),
            size => NonNull::new(unsafe { ffi::sn_rust_alloc_zeroed(layout.align(), size) }.cast()),
        }
    }

    /// Deallocates memory at the given pointer and layout.
    #[inline(always)]
    pub fn safe_dealloc(&self, ptr: *mut u8, layout: Layout) {
        match (ptr.is_null(), layout.size()) {
            (false, size) if size > 0 => unsafe {
                ffi::sn_rust_dealloc(ptr.cast(), layout.align(), size);
            },
            _ => {} // No action needed for null pointers or ZSTs.
        }
    }

    /// Reallocates memory at the given pointer and layout to a new size.
    #[inline(always)]
    pub fn safe_realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        new_size: usize,
    ) -> Option<NonNull<u8>> {
        match (layout.size(), new_size) {
            (0, 0) => Some(self.handle_zst()), // Both old and new sizes are zero.
            (0, _) => self.safe_alloc(Layout::from_size_align(new_size, layout.align()).ok()?),
            (_, 0) => {
                self.safe_dealloc(ptr, layout);
                None // New size is zero; deallocate and return None.
            }
            _ => NonNull::new(unsafe {
                ffi::sn_rust_realloc(ptr.cast(), layout.align(), layout.size(), new_size).cast()
            }),
        }
    }
    /// Allocates memory with the given layout, returning a non-null pointer on success
    #[inline(always)]
    pub fn alloc_aligned(&self, layout: Layout) -> Option<NonNull<u8>> {
        match layout.size() {
            0 => Some(self.handle_zst()),
            size => NonNull::new(unsafe { ffi::sn_rust_alloc(layout.align(), size) }.cast())
        }
    }
    /// Returns the usable size of an allocated block.
    #[inline(always)]
    pub fn usable_size(&self, ptr: *const u8) -> Option<usize> {
        match ptr.is_null() {
            true => None,
            false => Some(unsafe { ffi::sn_rust_usable_size(ptr.cast()) }),
        }
    }
}

unsafe impl GlobalAlloc for SnMalloc {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.safe_alloc(layout).map_or(ptr::null_mut(), |ptr| ptr.as_ptr())
    }

    #[inline(always)]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.safe_alloc_zeroed(layout)
            .map_or(ptr::null_mut(), |ptr| ptr.as_ptr())
    }

    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.safe_dealloc(ptr, layout);
    }

    #[inline(always)]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.safe_realloc(ptr, layout, new_size)
            .map_or(ptr::null_mut(), |ptr| ptr.as_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_lifecycle() {
        let alloc = SnMalloc::new();
        unsafe {
            let layout = Layout::from_size_align(8, 8).unwrap();
            
            // Test regular allocation
            let ptr = alloc.alloc(layout);
            alloc.dealloc(ptr, layout);

            // Test zeroed allocation
            let ptr = alloc.alloc_zeroed(layout);
            alloc.dealloc(ptr, layout);

            // Test reallocation
            let ptr = alloc.alloc(layout);
            let ptr = alloc.realloc(ptr, layout, 16);
            alloc.dealloc(ptr, layout);

            // Test large allocation
            let large_layout = Layout::from_size_align(1 << 20, 32).unwrap();
            let ptr = alloc.alloc(large_layout);
            alloc.dealloc(ptr, large_layout);
        }
    }
    #[test]
    fn it_frees_allocated_memory() {
        let alloc = SnMalloc::new();
        unsafe {
            let layout = Layout::from_size_align(8, 8).unwrap();
            let ptr = alloc.alloc(layout);
            alloc.dealloc(ptr, layout);
        }
    }

    #[test]
    fn it_frees_zero_allocated_memory() {
        let alloc = SnMalloc::new();
        unsafe {
            let layout = Layout::from_size_align(8, 8).unwrap();

            let ptr = alloc.alloc_zeroed(layout);
            alloc.dealloc(ptr, layout);
        }
    }

    #[test]
    fn it_frees_reallocated_memory() {
        let alloc = SnMalloc::new();
        unsafe {
            let layout = Layout::from_size_align(8, 8).unwrap();

            let ptr = alloc.alloc(layout);
            let ptr = alloc.realloc(ptr, layout, 16);
            alloc.dealloc(ptr, layout);
        }
    }

    #[test]
    fn it_frees_large_alloc() {
        let alloc = SnMalloc::new();
        unsafe {
            let layout = Layout::from_size_align(1 << 20, 32).unwrap();

            let ptr = alloc.alloc(layout);
            alloc.dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_usable_size() {
        let alloc = SnMalloc::new();
        unsafe {
            let layout = Layout::from_size_align(8, 8).unwrap();
            let ptr = alloc.alloc(layout);
            let usz = alloc.usable_size(ptr).expect("usable_size returned None");
            alloc.dealloc(ptr, layout);
            assert!(usz >= 8);
        }
    }
    
    #[test]
    fn test_zero_sized_allocation() {
        let alloc = SnMalloc::new();
        let zst_layout = Layout::from_size_align(0, 1).unwrap();

        unsafe {
            let ptr = alloc.alloc(zst_layout);
            assert!(!ptr.is_null());
            alloc.dealloc(ptr, zst_layout);
        }
    }

}
