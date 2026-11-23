//! Carries out the Zen 2 instructions an Intel host lacks.
//!
//! Titles are built for the console's Zen 2 and may use AMD-only
//! instructions; an Intel CPU raises an illegal-instruction fault on them,
//! which a vectored handler takes and emulates. On an AMD host they never
//! fault and none of this runs.
//!
//! * `clzero`: zeroes the 64-byte line holding `[rax]`. Memsets wrap it in
//!   `add rax, 64; cmp rax, r10; jb` back to it; that whole loop is run at
//!   once, not one fault per line.
//! * `monitorx` and `mwaitx`: arm a wait on a line, then wait for a store
//!   to it. `mwaitx` may wake for no reason and code re-checks after it, so
//!   both do nothing here; the fault itself is the pause.
//! * `extrq` and `insertq` (SSE4a): bit-field moves in an xmm register.

use windows_sys::Win32::Foundation::EXCEPTION_ILLEGAL_INSTRUCTION;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS,
};

const CONTINUE_EXECUTION: i32 = -1;
const CONTINUE_SEARCH: i32 = 0;
const LINE: u64 = 64;
const CLZERO: [u8; 3] = [0x0f, 0x01, 0xfc];
const MONITORX: [u8; 3] = [0x0f, 0x01, 0xfa];
const MWAITX: [u8; 3] = [0x0f, 0x01, 0xfb];
/// `add rax, 64; cmp rax, r10; jb` back to the `clzero`.
const CLZERO_LOOP: [u8; 9] = [0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4];

/// Installs the handler, once, ahead of the crash reporter's.
pub fn emulate_zen2() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only reads the faulting code and writes what the
    // instruction it emulates would have written.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes the exception being dispatched.
    let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    if record.ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION {
        return CONTINUE_SEARCH;
    }
    // shortcut: code is read only up to the end of the faulting page, so an
    // instruction split across two pages is not recognised; read the next
    // page too if one ever is.
    let page_left = 0x1000 - (context.Rip as usize & 0xfff);
    let mut bytes = [0u8; 16];
    let n = page_left.min(bytes.len());
    // SAFETY: on the page the CPU just fetched the faulting instruction from.
    unsafe { std::ptr::copy_nonoverlapping(context.Rip as *const u8, bytes.as_mut_ptr(), n) };
    let code = &bytes[..n];
    let len = if code.starts_with(&MONITORX) || code.starts_with(&MWAITX) {
        3
    } else if code.starts_with(&CLZERO) {
        // SAFETY: what the title's own instructions write.
        unsafe {
            if code[3..].starts_with(&CLZERO_LOOP) {
                clzero_loop(context);
                return CONTINUE_EXECUTION;
            }
            zero_lines(context.Rax, 1);
        }
        3
    } else if let Some(len) = sse4a(code, context) {
        len
    } else {
        return CONTINUE_SEARCH;
    };
    context.Rip += len as u64;
    CONTINUE_EXECUTION
}

/// `extrq` and `insertq`, register operands only (as SSE4a defines them):
/// moves a field of `length` bits (0 means 64) at bit `index` of an xmm
/// register's low quadword. The high quadword is left as it is (SSE4a
/// leaves it undefined). Returns the instruction's length.
fn sse4a(code: &[u8], c: &mut CONTEXT) -> Option<usize> {
    let (&prefix, rest) = code.split_first()?;
    let (rex, rest) = match rest.first() {
        Some(&r) if r & 0xf0 == 0x40 => (r, &rest[1..]),
        _ => (0, rest),
    };
    let &[0x0f, op @ (0x78 | 0x79), modrm, ref imm @ ..] = rest else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = usize::from((modrm >> 3) & 7 | (rex & 4) << 1);
    let rm = usize::from(modrm & 7 | (rex & 1) << 3);
    let len = code.len() - rest.len() + 3;
    let field = |length: u64, index: u64| (length & 63, index & 63);
    // SAFETY: an exception context holds the legacy floating-point state.
    let xmm = unsafe { &mut c.Anonymous.FltSave.XmmRegisters };
    match (prefix, op, imm) {
        (0x66, 0x78, &[length, index, ..]) if reg & 7 == 0 => {
            let f = field(length.into(), index.into());
            xmm[rm].Low = extract(xmm[rm].Low, f);
            Some(len + 2)
        }
        (0x66, 0x79, _) => {
            let s = xmm[rm].Low;
            xmm[reg].Low = extract(xmm[reg].Low, field(s, s >> 8));
            Some(len)
        }
        (0xf2, 0x78, &[length, index, ..]) => {
            let f = field(length.into(), index.into());
            xmm[reg].Low = insert(xmm[reg].Low, xmm[rm].Low, f);
            Some(len + 2)
        }
        (0xf2, 0x79, _) => {
            let s = xmm[rm].High as u64;
            xmm[reg].Low = insert(xmm[reg].Low, xmm[rm].Low, field(s, s >> 8));
            Some(len)
        }
        _ => None,
    }
}

fn mask(length: u64) -> u64 {
    if length == 0 {
        u64::MAX
    } else {
        (1 << length) - 1
    }
}

fn extract(v: u64, (length, index): (u64, u64)) -> u64 {
    (v >> index) & mask(length)
}

fn insert(d: u64, s: u64, (length, index): (u64, u64)) -> u64 {
    let m = u128::from(mask(length)) << index;
    ((u128::from(d) & !m) | ((u128::from(s) << index) & m)) as u64
}

/// # Safety
/// The `lines` lines from the one holding `at` are writable.
unsafe fn zero_lines(at: u64, lines: u64) {
    // SAFETY: forwarded.
    unsafe { std::ptr::write_bytes((at & !(LINE - 1)) as *mut u8, 0, (lines * LINE) as usize) };
}

/// Runs `do { clzero; rax += 64 } while rax < r10` and steps past it, with
/// the flags its last `cmp` leaves.
///
/// # Safety
/// As the title's loop: the lines it zeroes are writable.
unsafe fn clzero_loop(c: &mut CONTEXT) {
    let (start, end) = (c.Rax, c.R10);
    let lines = end.saturating_sub(start).div_ceil(LINE).max(1);
    // SAFETY: forwarded.
    unsafe { zero_lines(start, lines) };
    let rax = start.wrapping_add(lines * LINE);
    c.Rax = rax;
    c.EFlags = cmp_flags(c.EFlags, rax, end);
    c.Rip += (CLZERO.len() + CLZERO_LOOP.len()) as u64;
}

/// `eflags` with CF, PF, AF, ZF, SF and OF as `cmp a, b` sets them.
fn cmp_flags(eflags: u32, a: u64, b: u64) -> u32 {
    const CF: u32 = 1;
    const PF: u32 = 1 << 2;
    const AF: u32 = 1 << 4;
    const ZF: u32 = 1 << 6;
    const SF: u32 = 1 << 7;
    const OF: u32 = 1 << 11;
    let r = a.wrapping_sub(b);
    let mut f = eflags & !(CF | PF | AF | ZF | SF | OF);
    let set = |f: &mut u32, flag, on: bool| *f |= if on { flag } else { 0 };
    set(&mut f, CF, a < b);
    set(&mut f, PF, (r as u8).count_ones().is_multiple_of(2));
    set(&mut f, AF, (a ^ b ^ r) & 0x10 != 0);
    set(&mut f, ZF, r == 0);
    set(&mut f, SF, r >> 63 != 0);
    set(&mut f, OF, ((a ^ b) & (a ^ r)) >> 63 != 0);
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree,
    };

    /// Runs `code` as `fn(rcx, rdx) -> rax` from an executable page.
    fn run(code: &[u8], rcx: u64, rdx: u64) -> u64 {
        emulate_zen2();
        // SAFETY: a fresh page, filled with our code, then called and freed.
        unsafe {
            let page = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            );
            assert!(!page.is_null());
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
            let f: extern "win64" fn(u64, u64) -> u64 = std::mem::transmute(page);
            let rax = f(rcx, rdx);
            VirtualFree(page, 0, MEM_RELEASE);
            rax
        }
    }

    const MOV_RAX_RCX: [u8; 3] = [0x48, 0x89, 0xc8];
    const MOV_R10_RDX: [u8; 3] = [0x49, 0x89, 0xd2];
    const RET: u8 = 0xc3;

    #[repr(C, align(64))]
    struct Lines([u8; 64 * 8]);

    #[test]
    fn clzero_zeroes_the_line_holding_rax() {
        let mut buf = Lines([0xaa; 64 * 8]);
        let base = buf.0.as_mut_ptr() as u64;
        let code = [&MOV_RAX_RCX[..], &CLZERO, &[RET]].concat();
        assert_eq!(run(&code, base + 64 * 2 + 17, 0), base + 64 * 2 + 17);
        let zeroed: Vec<usize> = (0..8)
            .filter(|&l| buf.0[l * 64..][..64] == [0; 64])
            .collect();
        assert_eq!(zeroed, [2]);
        assert!(buf.0[64 * 3..].iter().all(|&b| b == 0xaa));
    }

    #[test]
    fn the_memset_loop_runs_whole() {
        let mut buf = Lines([0xaa; 64 * 8]);
        let base = buf.0.as_mut_ptr() as u64;
        let code = [
            &MOV_RAX_RCX[..],
            &MOV_R10_RDX,
            &CLZERO,
            &CLZERO_LOOP,
            &[RET],
        ]
        .concat();
        let rax = run(&code, base + 64, base + 64 * 5 - 8);
        assert_eq!(rax, base + 64 * 5);
        let zeroed: Vec<usize> = (0..8)
            .filter(|&l| buf.0[l * 64..][..64] == [0; 64])
            .collect();
        assert_eq!(zeroed, [1, 2, 3, 4]);
    }

    #[test]
    fn monitorx_and_mwaitx_return() {
        let code = [&MOV_RAX_RCX[..], &MONITORX, &MWAITX, &[RET]].concat();
        let line = Lines([0; 64 * 8]);
        let at = line.0.as_ptr() as u64;
        assert_eq!(run(&code, at, 0), at);
    }

    #[test]
    fn sse4a_field_moves() {
        const MOVQ_XMM0_RCX: [u8; 5] = [0x66, 0x48, 0x0f, 0x6e, 0xc1];
        const MOVQ_XMM1_RDX: [u8; 5] = [0x66, 0x48, 0x0f, 0x6e, 0xca];
        const MOVQ_XMM9_RDX: [u8; 5] = [0x66, 0x4c, 0x0f, 0x6e, 0xca];
        const MOVQ_RAX_XMM0: [u8; 5] = [0x66, 0x48, 0x0f, 0x7e, 0xc0];
        let with = |op: &[u8], load: &[u8]| {
            [&MOVQ_XMM0_RCX[..], load, op, &MOVQ_RAX_XMM0, &[RET]].concat()
        };
        let (d, s) = (0x1122_3344_5566_7788, 0xaaaa_bbbb_cccc_dddd);
        // insertq xmm0, xmm1, 16, 16
        let code = with(&[0xf2, 0x0f, 0x78, 0xc1, 16, 16], &MOVQ_XMM1_RDX);
        assert_eq!(run(&code, d, s), 0x1122_3344_dddd_7788);
        // insertq xmm0, xmm9, 0, 0: all 64 bits
        let code = with(&[0xf2, 0x41, 0x0f, 0x78, 0xc1, 0, 0], &MOVQ_XMM9_RDX);
        assert_eq!(run(&code, d, s), s);
        // extrq xmm0, 8, 4
        let code = with(&[0x66, 0x0f, 0x78, 0xc0, 8, 4], &[]);
        assert_eq!(run(&code, d, 0), 0x78);
        // extrq xmm0, xmm1, with xmm1 = index 12 << 8 | length 20
        let code = with(&[0x66, 0x0f, 0x79, 0xc1], &MOVQ_XMM1_RDX);
        assert_eq!(run(&code, d, 12 << 8 | 20), 0x5_5667);
    }

    #[test]
    fn flags_follow_cmp() {
        // ZF on equal, CF and SF when below, OF on signed overflow.
        assert_eq!(cmp_flags(0, 5, 5) & (1 | 1 << 6), 1 << 6);
        assert_eq!(cmp_flags(0, 4, 5) & (1 | 1 << 7), 1 | 1 << 7);
        assert_ne!(cmp_flags(0, 1 << 63, 1) & 1 << 11, 0);
        assert_eq!(cmp_flags(0x202, 9, 1) & 0x200, 0x200, "other flags kept");
    }
}
