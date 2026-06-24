# 001 · Ascend codegen — emit integer `max`/`min` as a ternary

| | |
|---|---|
| **Compiler repo** | `yangqiang2018/tilelang-ascend-2` |
| **Branch / commit** | `wip/ascend-codegen-int-minmax-ternary` · `0e53a8ad` (off `ascendc_pto`) |
| **Files changed** | `src/target/codegen_ascend.h`, `src/target/codegen_ascend.cc` |
| **Necessary?** | Yes — the SWA kernel does not compile without it |
| **Compatible?** | Yes — verified by regression (see §6) |
| **Status** | Verified, pending merge to `ascendc_pto` |

---

## 1. Why the operator needs scalar integer `max`/`min`

The faithful SWA (sliding-window attention) port computes, per query token, the
window bounds and a clamped paged-gather position — exactly mirroring the
Ascend C `oriMaskLeft`/`oriMaskRight` math and `DataCopyPA` window load:

```python
ori_left  = T.max(s_global - ori_win_left, 0)       # clamp window start to >= 0
ori_right = s_global + 1
pos       = T.min(ori_left + row, ori_right - 1)    # clamp over-gathered rows
page      = ori_block_table[b, pos // ori_block_size]
```

These are **scalar integer index computations**. They are required for the
windowed paged KV gather and are the direct TileLang spelling of the Ascend C
`Max(...)` / `Min(...)` helpers (`sparse_attn_sharedkv_swa_kernel.h:689-693`,
`common.h`). They are not optional or stylistic.

## 2. Symptom

After TileLang codegen succeeded, `bisheng` (the Ascend C++ compiler) rejected
the generated kernel:

```
/tmp/tmpXXXX.cpp:112:67: error: call to 'max' is ambiguous
```

The offending generated line (from `func.get_kernel_source()`):

```cpp
// act_q, act_kv are int32_t; cid (the grid block var) is int64
int32_t page = ori_block_table.GetValue(
    ( min( ((vid*64) + max((((act_kv + cid) - act_q) - 127), 0)) + r,
           ((act_kv + cid) - act_q) ) / 128 ) );
```

`max((((act_kv + cid) - act_q) - 127), 0)` is `max(int64_t, int)` — a mixed
integer-width call. In the bisheng / CANN C++ environment `max` resolves to
multiple candidates and the mixed widths make the call **ambiguous**.

## 3. Root cause

`CodeGenTileLangAscend` (the A2/A3 `is_npu` Ascend backend,
`src/target/codegen_ascend.cc`) inherited `CodeGenC`'s default emission for
`MaxNode` / `MinNode`, which prints a **bare** `max(a, b)` / `min(a, b)`. With
mixed integer widths (an `int64` grid var vs an `int` literal) that unqualified
name is ambiguous to bisheng.

- The PTO backend (`codegen_ascend_pto.cc`) uses `std::max`; the non-PTO path
  had **no `Max`/`Min` override** at all.
- No existing example kernel emits a *runtime scalar integer* `max`/`min` in a
  kernel body — their `T.max`/`T.min` are over compile-time constants and get
  folded away (e.g. `n_num = T.max(T.ceildiv(...), 1)` in
  `paged_flash_attn_bhsd.py`). So this codegen gap was **latent / untested**
  until this operator became the first to need it.

## 4. Why it could not be fixed in the kernel

Attempted kernel-side fix: cast the indices to `int32` so the operands match
(`s = T.cast(cid % max_seq, "int32")`). It **did not work** — TVM's arithmetic
simplifier folds `cast(cid % max_seq, int32)` → `cid` (it proves
`cid % max_seq == cid` for `cid < max_seq` and drops the narrowing cast), so the
`int64` type survives into codegen and the cast never reaches the generated C++.

Expressing `max(x, 0)` via `T.if_then_else` instead risks the simplifier
canonicalizing the resulting `Select` back into a `Max` node, re-emitting the
same ambiguous call. There is no reliable way for the kernel to avoid producing
an integer `Max`/`Min` IR node, so **codegen is the correct layer to fix**.

## 5. The fix

Override `VisitExpr_(MaxNode)` / `VisitExpr_(MinNode)` in
`CodeGenTileLangAscend` to emit a **ternary** for integer/uint dtypes; float and
other dtypes fall through to the unchanged base implementation.

`src/target/codegen_ascend.h` (declarations, next to the existing FloorDiv/Mod
overrides):

```cpp
void VisitExpr_(const MaxNode *op, std::ostream &os);
void VisitExpr_(const MinNode *op, std::ostream &os);
```

`src/target/codegen_ascend.cc`:

```cpp
void CodeGenTileLangAscend::VisitExpr_(const MaxNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    os << "("; PrintExpr(op->a, os); os << " > "; PrintExpr(op->b, os);
    os << " ? "; PrintExpr(op->a, os); os << " : "; PrintExpr(op->b, os);
    os << ")";
  } else {
    CodeGenC::VisitExpr_(op, os);   // float etc.: unchanged (NaN semantics)
  }
}
// MinNode: same, with `<` instead of `>`.
```

After the fix the generated line becomes overload-free:

```cpp
(((act_kv + cid) - act_q) - 127) > 0 ? (((act_kv + cid) - act_q) - 127) : 0
```

## 6. Why it is compatibility-preserving (and the proof)

Design:
- **Scoped to integer/uint `Max`/`Min` only.** Float and every other dtype use
  the unchanged `CodeGenC` path → NaN/float semantics are untouched.
- A ternary is **semantically identical** to integer max/min and needs **no
  overload resolution**, so it is strictly more robust than the bare call.
- No existing operator emits a scalar integer `Max`/`Min` in a kernel body
  (examples fold them at compile time), so existing ops' output is unchanged.

Empirical proof (after rebuilding the compiler from source with this change on
the NPU container):
- `examples/flash_attention/paged_flash_attn_bhsd.py` → **`Kernel Output Match!`**
- `examples/developer_mode/sparse_flash_attn_developer.py` → **`Test Passed!`**

Both regression kernels go through the same `CodeGenTileLangAscend`; their
passing confirms the change does not break other operators.

## 7. Faithfulness note

This honours the project rule *"编译器有 bug→修编译器；所有对编译器的修改都必须是
兼容性修改"*. It is a genuine codegen bug (ambiguous emission of a standard
operation), fixed minimally and compatibly. The operator uses ordinary
`max`/`min` — the same operation as the Ascend C reference's `Max()`/`Min()`;
the fix only changes how the codegen *prints* them. No new method was invented
and no kernel-side workaround was used.
