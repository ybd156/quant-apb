# 28 · 训练机配对：授权链接 + 授权码

> 代码：`qcore/pairing.py`　接口：`POST /api/pair/code`（生成）、`POST /api/pair/redeem`（兑换）、`GET /api/pair`（状态）
> 训练机侧：`python tools/train_machine_auto.py --pair "<链接或码>"`　测试：`tests/test_pairing.py`（20/20）

---

## 一、它替掉的是什么

原来给训练机配置要三步手工操作：

1. NAS「设置 → 访问令牌」新建令牌（**明文只显示一次**）；
2. 抄下 NAS 地址 + 那串 `apb_...`；
3. 到训练机 `APB.bat auto` 里粘两遍。

抄错一位、忘了复制、令牌作废 —— 都会得到一句 `login_required`，看不出是配置问题。

现在：**NAS 上点一下「生成授权链接」→ 训练机粘一条命令**。

---

## 二、两条命令

**NAS 侧**（网页「设置 → 训练机配对」按钮，或命令行）：

```bash
python -m qcore.pairing --new --nas=https://apbcdn.683868.xyz
# → 授权码 BE67-4DZ2 · 链接 https://apbcdn.683868.xyz/#pair=BE67-4DZ2 · 30 分钟有效
```

**训练机侧**（三种写法都认）：

```bash
python tools/train_machine_auto.py --pair "https://apbcdn.683868.xyz/#pair=BE67-4DZ2"
python tools/train_machine_auto.py --pair BE67-4DZ2          # 需先有 NAS 地址
python tools/train_machine_auto.py --pair "apb://pair?nas=https://apbcdn.683868.xyz&code=BE67-4DZ2"
```

兑换成功后：`apb_defaults.json` 写好（`nas` + 新令牌，权限 600）→ 自动接着走
「连通检查 → 起后台守护」，和 `auto` 是同一条路径。

---

## 三、安全边界（写清楚，别当它是密码）

| 约束 | 取值 | 说明 |
|---|---|---|
| 一次性 | 兑换成功即 `used_at` | 第二次直接 `already_used`，要重新生成 |
| 有效期 | 默认 **30 分钟** | 过期即不可用；`create(ttl_min=...)` 可调 |
| 字符集 | 去掉 `0/O/1/I/L` 的 30 个字符 | 手抄/口述不会混 |
| 长度 | 8 位 → 30⁸ ≈ 6.5×10¹¹ | 配合 TTL + 单次 + 限速，猜中概率可忽略 |
| 限速 | 同一 IP 每分钟最多 **10 次失败** | 防暴力猜码；别的 IP 不受影响 |
| 令牌 | 兑换时才签发 | 32 字节随机串，可单独吊销（`/api/tokens`） |
| 存储 | `data/state/pair_codes.json` | 只存码与状态，不存令牌明文 |

**它仍然是一张「票」**：谁拿到未过期的码，谁就能换到一个 `sync+write` 令牌。
所以别发到公开频道 —— 这一点在 API 返回的 `note` 里也写了。

---

## 四、边界与失败语义

| 情况 | 返回 | 提示 |
|---|---|---|
| 码不存在 | `unknown_code` | 拼写错误或已被清理 |
| 已用过 | `already_used`（带 `used_at`） | 一次性，请重新生成 |
| 过期 | `expired`（带 `expires_at`） | 同上 |
| 格式不对 | `bad_code` | 应为 8 位（形如 `K7M2-QP4R`） |
| 刷码太快 | `rate_limited` | 每分钟 10 次失败上限 |
| 只有码没地址 | 训练机侧 `need_nas` | 用完整链接，或先填一次 NAS 地址 |

---

## 五、验收

```
tests/test_pairing.py  20/20
  · 码 8 位、无易混字符
  · 链接三形态（#pair= / ?pair= / apb://）都能解析回 (nas, code)
  · 裸码容忍 小写/空格/缺横线
  · 兑换 → 签发令牌（tokens.verify() 认得）→ 二次兑换被拒
  · 过期码被拒 · 乱码被拒 · 连续失败被限速 · 别的 IP 不受影响
```

HTTP 端到端（本机起服务实测）：

```
POST /api/pair/code   {"nas":"http://127.0.0.1:8796"}  → code + 三种链接
POST /api/pair/redeem {"code":"KC2B-VBCE"}             → apb_... 令牌（无需预先鉴权）
POST /api/pair/redeem {"code":"KC2B-VBCE"} 再来一次     → already_used
```

训练机侧 `--pair "<链接>"` 实测写入：

```json
{"nas":"http://127.0.0.1:8796","token":"apb_…","paired_at":"…","paired_code":"KC2B-VBCE"}
```

---

## 六、为什么兑换接口不要求先鉴权

它是**唯一**的"没有令牌但需要令牌"的入口 —— 码本身承担凭据角色。
用令牌保护它就等于"要先有令牌才能拿令牌"。

代价是它暴露在公网，所以：单次使用 + 30 分钟 TTL + 30⁸ 空间 + 每分钟 10 次失败限速。
真正的兜底是 TTL：漏出去的码最多活半小时，且只能换一个可吊销的令牌。
