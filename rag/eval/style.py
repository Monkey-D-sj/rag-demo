"""终端输出样式工具。

注意：_disp_width / _pad 对 CJK 字符串使用特殊的显示宽度计算，
中文环境下的表格对齐依赖它们，不要用普通字符串对齐替代。
"""

import re

# ── 基础样式 ──

_RESET = "\033[0m"


def bold(s: str) -> str:
    """粗体"""
    return f"\033[1m{s}{_RESET}"


def dim(s: str) -> str:
    """暗色 / 次要信息"""
    return f"\033[2m{s}{_RESET}"


# ── 语义颜色 ──

def green(s: str) -> str:
    """正向指标：涨、通过、最佳"""
    return f"\033[32m{s}{_RESET}"


def red(s: str) -> str:
    """负向指标：跌、失败、最差"""
    return f"\033[31m{s}{_RESET}"


def yellow(s: str) -> str:
    """警告"""
    return f"\033[33m{s}{_RESET}"


def cyan(s: str) -> str:
    """信息 / 阶段标题"""
    return f"\033[36m{s}{_RESET}"


# ── 组合样式 ──

def success(s: str) -> str:
    """✅ 成功"""
    return green(bold(f"✅ {s}"))


def failure(s: str) -> str:
    """❌ 失败"""
    return red(bold(f"❌ {s}"))


def phase(s: str) -> str:
    """▶ 阶段标题"""
    return bold(cyan(f"▶ {s}"))


# ── 评测专用 ──

def delta_str(delta: float, fmt: str = ".4f") -> str:
    """带颜色箭头的差值：▲+0.0234 (绿) / ▼-0.0150 (红)。"""
    if delta >= 0:
        return green(f"▲{delta:+{fmt}}")
    else:
        return red(f"▼{delta:{fmt}}")  # 自带负号


def drop_arrow(rel_drop: float, fmt: str = ".1%") -> str:
    """相对涨跌箭头：▲12.3% (涨/绿) 或 ▼5.0% (跌/红)。"""
    if rel_drop <= 0:
        return green(f"▲{abs(rel_drop):{fmt}}")
    else:
        return red(f"▼{rel_drop:{fmt}}")


def best_val(val: str) -> str:
    """行内最佳值（绿色粗体）"""
    return bold(green(val))


def worst_val(val: str) -> str:
    """行内最差值（红色）"""
    return red(val)


# ── 颜色剥离（写文件或管道时用）──

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip(s: str) -> str:
    """移除所有 ANSI 转义序列。"""
    return _ANSI_RE.sub("", s)
