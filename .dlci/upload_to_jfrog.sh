#!/bin/bash

# upload_to_jfrog.sh - JFrog Artifactory upload script
# Upload a wheel package to JFrog, auto-detecting the package name.
#
# SECURITY NOTE: unlike the upstream vLLM version, the JFrog access token is
# NEVER hardcoded here. It is read from the JFROG_TOKEN environment variable
# (set as a masked GitLab CI/CD variable). The script fails loudly if no token
# is available and JFrog CLI is not already configured.

set -e

# 参数
JFROG_REPO="${JFROG_SGLANG_REPO:-dl-pypi}"
WHL_PATH=""
VERBOSE=false
DRY_RUN=false

# 显示用法
usage() {
    echo "用法: $0 [选项] <wheel包路径>"
    echo ""
    echo "选项:"
    echo "  --repo REPO        JFrog 仓库名 (默认: dl-pypi / \$JFROG_SGLANG_REPO)"
    echo "  --dry-run          仅显示操作，不实际上传"
    echo "  --verbose          详细输出"
    echo "  --help             显示帮助"
    echo ""
    echo "环境变量:"
    echo "  JFROG_TOKEN        JFrog 访问令牌 (必填，除非 jf CLI 已配置)"
    echo "  JFROG_URL          JFrog 服务地址 (默认: http://ext-artifactory.denglin.com:8082/)"
    echo ""
    echo "示例:"
    echo "  $0 sglang-0.5.16.dev560+g692b5c460.sdk202607101443-py3-none-any.whl"
    echo "  $0 --repo dl-pypi --verbose pytorch-2.3.0.whl"
    exit 1
}

# 日志函数
info() { echo "[信息] $*" >&2; }
error() { echo "[错误] $*" >&2; exit 1; }
verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "[详细] $*" >&2
    fi
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo) JFROG_REPO="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help) usage ;;
        -*) error "未知选项: $1" ;;
        *) WHL_PATH="$1"; shift ;;
    esac
done

# 验证参数
[ -z "$WHL_PATH" ] && error "缺少 wheel 包路径"
[ ! -f "$WHL_PATH" ] && error "wheel 包不存在: $WHL_PATH"

# 检查 jf 命令
if ! command -v jf >/dev/null 2>&1; then
    error "未找到 jf 命令，请安装 JFrog CLI"
fi

# 配置 JFrog CLI 认证 (token 从环境变量读取，绝不硬编码)
configure_jfrog_cli() {
    info "配置 JFrog CLI 认证..."

    JFROG_URL="${JFROG_URL:-http://ext-artifactory.denglin.com:8082/}"

    # 检查是否已经配置
    if jf config show 2>/dev/null | grep -q "denglin\|url"; then
        verbose "JFrog CLI 已配置，跳过认证配置"
        return 0
    fi

    # 必须有 token 才能配置
    if [ -z "${JFROG_TOKEN:-}" ]; then
        error "JFROG_TOKEN 环境变量未设置，且 JFrog CLI 未配置。请在 GitLab CI/CD 变量中设置 JFROG_TOKEN (masked)。"
    fi

    verbose "添加 JFrog CLI 服务器配置..."
    if jf config add my-server --url "$JFROG_URL" --access-token "$JFROG_TOKEN" --interactive=false 2>/dev/null; then
        verbose "✓ JFrog CLI 认证配置成功"
    else
        error "✗ JFrog CLI 认证配置失败 (JFROG_TOKEN 无效或网络不可达)"
    fi
}

# 转换为绝对路径
WHL_PATH=$(realpath "$WHL_PATH")
WHL_NAME=$(basename "$WHL_PATH")

info "准备上传: $WHL_NAME"

# 配置 JFrog CLI 认证（dry-run 模式下跳过）
if [ "$DRY_RUN" = false ]; then
    configure_jfrog_cli
else
    info "Dry-run 模式，跳过 JFrog CLI 认证配置"
fi

# 提取包名（从文件名中提取）
# 支持格式: package-version-py_tag-abi-platform.whl
PKG_NAME=$(echo "$WHL_NAME" | cut -d'-' -f1)
if [ -z "$PKG_NAME" ]; then
    error "无法从文件名提取包名: $WHL_NAME"
fi
verbose "提取到包名: $PKG_NAME"

# 生成目标路径
TARGET_PATH="${JFROG_REPO}/${PKG_NAME}/"
verbose "目标路径: $TARGET_PATH"

# 生成下载链接
DOWNLOAD_URL="http://ext-artifactory.denglin.com:8082/artifactory/${TARGET_PATH}${WHL_NAME}"

# 显示上传信息
info "包名: $PKG_NAME"
info "源文件: $WHL_PATH"
info "目标路径: $TARGET_PATH"
info "下载链接: $DOWNLOAD_URL"

# 执行上传或模拟
if [ "$DRY_RUN" = true ]; then
    info "模拟执行上传命令:"
    echo "jf rt u \"$WHL_PATH\" \"$TARGET_PATH\" --flat=true"
    echo ""
    echo "=========================================="
    echo "📦 Dry-run 模式 - 上传命令已生成"
    echo "=========================================="
    echo "包名: $PKG_NAME"
    echo "文件: $WHL_NAME"
    echo "模拟下载链接: $DOWNLOAD_URL"
    echo "=========================================="
    info "✓ Dry-run 完成"
    exit 0
else
    info "开始上传..."

    # 执行上传
    if jf rt u "$WHL_PATH" "$TARGET_PATH" --flat=true; then
        info "✓ 上传成功"
        echo ""
        echo "=========================================="
        echo "📦 Wheel 包上传完成"
        echo "=========================================="
        echo "包名: $PKG_NAME"
        echo "文件: $WHL_NAME"
        echo "下载链接: $DOWNLOAD_URL"
        echo "=========================================="
        echo ""

        # 输出下载链接（供脚本捕获）
        echo "$DOWNLOAD_URL"
        exit 0
    else
        error "✗ 上传失败"
    fi
fi
