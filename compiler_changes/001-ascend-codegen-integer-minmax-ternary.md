# 001 · Ascend codegen —— 整数 `max`/`min` 输出为三元表达式

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/ascend-codegen-int-minmax-ternary` · `0e53a8ad`（基于 `ascendc_pto`） |
| **改动文件** | `src/target/codegen_ascend.h`、`src/target/codegen_ascend.cc` |
| **是否必须** | 是 —— 不改的话 SWA 内核根本编译不过 |
| **是否兼容** | 是 —— 已通过回归验证(见 §6) |
| **状态** | 已验证,待合入 `ascendc_pto` |

---

## 1. 算子为什么需要标量整数 `max`/`min`

忠实复刻的 SWA(滑动窗口注意力)在每个 query token 上计算窗口边界和一个被钳制的
分页 gather 位置 —— 这与 Ascend C 的 `oriMaskLeft`/`oriMaskRight` 数学、以及
`DataCopyPA` 的窗口加载是一一对应的:

```python
ori_left  = T.max(s_global - ori_win_left, 0)       # 窗口左界钳到 >= 0
ori_right = s_global + 1
pos       = T.min(ori_left + row, ori_right - 1)    # 越界行钳到窗口末位
page      = ori_block_table[b, pos // ori_block_size]
```

这些都是**标量整数下标计算**,是滑动窗口分页 KV gather 所必需的,也是 Ascend C
`Max(...)` / `Min(...)`(`sparse_attn_sharedkv_swa_kernel.h:689-693`、`common.h`）
的直接 TileLang 写法,不是可选项、也不是风格问题。

## 2. 现象

TileLang codegen 成功之后,`bisheng`(Ascend C++ 编译器)拒绝编译生成的内核:

```
/tmp/tmpXXXX.cpp:112:67: error: call to 'max' is ambiguous
```

出问题的生成代码(来自 `func.get_kernel_source()`):

```cpp
// act_q、act_kv 是 int32_t;cid(grid 块变量)是 int64
int32_t page = ori_block_table.GetValue(
    ( min( ((vid*64) + max((((act_kv + cid) - act_q) - 127), 0)) + r,
           ((act_kv + cid) - act_q) ) / 128 ) );
```

`max((((act_kv + cid) - act_q) - 127), 0)` 是 `max(int64_t, int)` —— 整数宽度
不一致。在 bisheng / CANN 的 C++ 环境里 `max` 有多个候选重载,宽度不一致使该调用
**有歧义**。

## 3. 根因

`CodeGenTileLangAscend`(A2/A3 的 `is_npu` Ascend 后端,
`src/target/codegen_ascend.cc`)继承了 `CodeGenC` 对 `MaxNode` / `MinNode` 的默认
输出 —— 打印**裸的** `max(a, b)` / `min(a, b)`。当整数宽度不一致(int64 grid 变量
对 int 字面量)时,这个未限定名字对 bisheng 有歧义。

- PTO 后端(`codegen_ascend_pto.cc`)用的是 `std::max`;而非 PTO 这条路**完全没有**
  `Max`/`Min` 的 override。
- 现有示例内核**从不**在 kernel body 里输出运行期标量整数 `max`/`min` —— 它们的
  `T.max`/`T.min` 都作用在编译期常量上、会被折叠掉(例如 `paged_flash_attn_bhsd.py`
  里的 `n_num = T.max(T.ceildiv(...), 1)`)。所以这个 codegen 缺口一直**潜伏、未被
  覆盖**,直到本算子成为第一个真正需要它的算子。

## 4. 为什么不能在内核侧解决

试过的内核侧改法:把下标 cast 成 `int32` 让两个操作数同类型
(`s = T.cast(cid % max_seq, "int32")`)。**无效** —— TVM 的算术 simplifier 会把
`cast(cid % max_seq, int32)` 折叠成 `cid`(它能证明 `cid < max_seq` 时
`cid % max_seq == cid`,从而丢掉这个收窄 cast),于是 `int64` 类型一路保留到 codegen,
cast 根本到不了生成的 C++。

改用 `T.if_then_else` 来写 `max(x, 0)` 又有风险:simplifier 可能把得到的 `Select`
反向规约回 `Max` 节点,重新输出同样有歧义的调用。所以内核没有可靠办法避免产生整数
`Max`/`Min` 的 IR 节点,**codegen 才是正确的修复层**。

## 5. 修法

在 `CodeGenTileLangAscend` 里 override `VisitExpr_(MaxNode)` /
`VisitExpr_(MinNode)`,对整数/uint 类型输出**三元表达式**;浮点及其它类型继续走未改动
的基类实现。

`src/target/codegen_ascend.h`(声明,放在已有的 FloorDiv/Mod override 旁边):

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
    CodeGenC::VisitExpr_(op, os);   // 浮点等:不改动(保留 NaN 语义)
  }
}
// MinNode:同上,把 `>` 换成 `<`。
```

修复后那一行生成代码变成无需重载决议的形式:

```cpp
(((act_kv + cid) - act_q) - 127) > 0 ? (((act_kv + cid) - act_q) - 127) : 0
```

## 6. 为什么它是兼容性修改(及证据)

设计上:
- **只作用于整数/uint 的 `Max`/`Min`。** 浮点及其它任何类型走未改动的 `CodeGenC`
  路径 → NaN/浮点语义完全不变。
- 三元表达式对整数 max/min 在语义上完全等价,且**无需重载决议**,因此严格比裸调用
  更健壮。
- 现有算子没有在 kernel body 里输出标量整数 `Max`/`Min`(示例都在编译期折叠),所以
  现有算子的输出完全不变。

实测证据(在 NPU 容器上带此修改从源码重编译器后):
- `examples/flash_attention/paged_flash_attn_bhsd.py` → **`Kernel Output Match!`**
- `examples/developer_mode/sparse_flash_attn_developer.py` → **`Test Passed!`**

两个回归内核都走同一个 `CodeGenTileLangAscend`;它们仍然通过,证明此改动不会破坏其它
算子。

## 7. 忠实性说明

这符合本项目规则 *“编译器有 bug→修编译器;所有对编译器的修改都必须是兼容性修改”*。
这是一个真实的 codegen bug(对一个标准操作输出了有歧义的代码),以最小且兼容的方式修复。
算子用的是普通的 `max`/`min` —— 与 Ascend C 参考实现里的 `Max()`/`Min()` 是同一个操作;
修复只改变 codegen *打印* 它们的方式。没有发明新方法,也没有用内核侧绕路。

## 8. 必要性与通用性

**必要性(为什么非改不可)。** SWA 的窗口边界与分页 gather 位置是**运行期标量整数**
`max`/`min`(`max(s_global-127,0)`、`min(ori_left+row, ori_right-1)`),与 Ascend C 的
`Max()`/`Min()` 一一对应,不是可选写法。codegen 对整数 `Max`/`Min` 输出**裸 `max(a,b)`**,
在 int64(grid 变量)对 int 字面量时是有歧义重载,bisheng **直接编译失败**。内核侧又躲不
掉(§4:simplifier 把收窄 cast 折掉)。所以**不改这一处,SWA 内核根本编不过**——这是
阻断性的、必须在 codegen 修。

**通用性(不止本算子)。** 这是一个**通用的 codegen 正确性修复**,与 sparse_attn_sharedkv
无关:**任何**会产生「**运行期**(不能被常量折叠的)标量整数 `max`/`min`」的算子,都会撞到
同一个歧义重载而编译失败。此前非 PTO 的 Ascend codegen **根本没有** `Max`/`Min` 的
override(只有 PTO 路径用 `std::max`),这个缺口一直**潜伏**——既有示例的 `T.max`/`T.min`
都作用在编译期常量上、会被折叠(如 `T.max(ceildiv(...),1)`),本算子只是**第一个**真正需要
运行期整数 min/max 的。修复(整数/uint 走三元、浮点走原路)是 dtype 通用的,任何后续做
下标/窗口/钳位运算的算子都直接受益。
