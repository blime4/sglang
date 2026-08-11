# 需求文档：c-41 loongarch64 CI 镜像增补科学计算编译工具链

| 项 | 内容 |
|---|---|
| 提出人 | shaobo.xie |
| 日期 | 2026-08-11 |
| 优先级 | 中（loong CI smoke 当前 `allow_failure`，不阻断 MR；修复后可转为强制门禁） |
| 交付对象 | ci-docker-images `c-41` 镜像维护者 |
| 相关分支 | `dl-loongarch64-outlines-rust`（dl 远程） |
| 数据来源 | 已在 c-41 容器内实测（OpenCloudOS Stream 23，AppStream + BaseOS 仓库） |

---

## 1. 背景

`compile_test_sglang_loongarch64` 这个 CI job：
- **编译阶段（build）已经是绿的** —— sglang wheel + sgl-kernel(dlcc) 都能正常产出。
- **smoke 阶段失败**：执行 `pip install sglang[srt_dl]` 会拉一整套科学计算依赖，这些包**在 `dl-virtual-loongarch64` 里没有 loongarch64 预编译 wheel**，pip/uv 只能退回**源码编译**，而当前 `c-41:manylinux_2_38-loongarch64` 镜像缺源码编译所需的本地工具链 → 逐个失败 → `import sglang` 报 `ModuleNotFoundError`。

x86_64 / aarch64 有预编译 wheel 不受影响，**只需改 loong 的 c-41 镜像**。

## 2. 影响范围（要改哪些镜像）

`scripts/run/config_env.json` → `docker_images` 里 loongarch64 引用两个 tag（同一镜像名 `c-41:manylinux_2_38-loongarch64`）：

| 用途 | 当前 tag | 是否必须含工具链 |
|---|---|---|
| build（编译） | `c-41:manylinux_2_38-loongarch64-20260629` | 是 |
| runtime / smoke（**scipy 在这里编译**） | `c-41:manylinux_2_38-loongarch64-20260612` | **必须** |

> 若是同一 Dockerfile 产出，打一个新 tag 即可；若分别构建，两者都要带。
> **只改 loong（c-41），不要动 c-31(amd64) / c-30(aarch64)。**

## 3. 需求清单（仅列龙芯源 OpenCloudOS 23 仓库里**确实存在**的包 + 实测版本）

| 包（实测版本） | 仓库 | 用途（被哪个 Python 依赖的源码编译需要） |
|---|---|---|
| `rust` `cargo`（1.96.0, `1.96.0-1.ocs23`） | BaseOS | outlines-core（outlines）、tiktoken —— Rust 扩展 |
| `gcc-gfortran`（12.3.1, `12.3.1.5-3.ocs23`） | BaseOS | scipy —— Fortran 编译 |
| `openblas-devel`（0.3.26, `0.3.26-6.ocs23`） | AppStream | scipy —— BLAS（**见第 4 节，需补 `.pc`**） |
| `cmake`（3.26.5, `3.26.5-7.ocs23`） | BaseOS | xgrammar —— C++ 构建 |
| `pkgconfig`（即 `pkgconf-pkg-config` 1.9.5） | BaseOS | scipy/xgrammar 通过 pkg-config 找库 |
| `zlib-devel`（1.2.13） | BaseOS | pillow |
| `libjpeg-turbo-devel` | BaseOS | pillow（JPEG） |
| `freetype-devel`（2.13.1） | BaseOS | pillow（字体） |
| `libtiff-devel` | BaseOS | pillow（TIFF） |
| `libwebp-devel`（1.3.2） | BaseOS | pillow（WebP） |

**仓库里没有、本次不要求的：**
- `ffmpeg-devel` —— **OpenCloudOS 23 仓库内没有**（在 EPEL/RPM Fusion）。所以 `av`/PyAV 在 loong 上**不构建**，已用 `python/pyproject_dl.toml` 的 `av ; platform_machine != 'loongarch64'` 平台排除（av 是视频/多模态，smoke 不涉及）。loong 要支持视频再说。

参考 Dockerfile 片段：

```dockerfile
RUN dnf install -y --setopt=install_weak_deps=False \
      rust cargo \
      gcc-gfortran \
      openblas-devel \
      cmake pkgconfig \
      zlib-devel libjpeg-turbo-devel freetype-devel libtiff-devel libwebp-devel \
 && dnf clean all
```

## 4. 关键技术要求：手动生成 `openblas.pc` ⚠️（这是当前卡住 scipy 的点）

实测结论：OpenCloudOS 23 的 `openblas-devel`（0.3.26）**只装了 `.so` 库**（`/usr/lib64/libopenblas*.so`），**不带 pkg-config 的 `.pc` 文件**。系统里虽有 `blas.pc`/`lapack.pc`（参考 BLAS），但 scipy 1.18 的 meson **只认 `openblas`**（`dependency('openblas')`，找不到就报 `Dependency "OpenBLAS" not found` 直接停），不会回退用参考 BLAS。

**要求**：镜像里手动生成 `/usr/lib64/pkgconfig/openblas.pc`，指向已安装的 `libopenblas.so`。Dockerfile 加一段：

```dockerfile
# openblas-devel 不带 .pc，scipy 的 meson 需要通过 pkg-config 找到 openblas，这里补一个
RUN printf '%s\n' \
    'prefix=/usr' \
    'exec_prefix=${prefix}' \
    'libdir=${exec_prefix}/lib64' \
    'includedir=${prefix}/include' \
    '' \
    'Name: OpenBLAS' \
    'Description: OpenBLAS reference implementation' \
    'Version: 0.3.26' \
    'Libs: -L${libdir} -lopenblas' \
    'Cflags: -I${includedir}' \
    > /usr/lib64/pkgconfig/openblas.pc
```

> 如果实际 `libopenblas.so` 的名字/路径不同，按 `rpm -ql openblas-devel | grep '\.so$'` 的结果调整 `-lopenblas` 与 `libdir`。

## 5. 验收标准

镜像构建完后，在一个**全新 `docker run`** 的容器里验证（全部通过即合格）：

```bash
# (1) 工具链就位
rustc --version && cargo --version && gfortran --version && cmake --version

# (2) openblas 可被 pkg-config 找到（关键）
pkg-config --exists openblas && echo "openblas OK" || echo "openblas MISSING"
pkg-config --modversion openblas   # 应打印 0.3.26
pkg-config --libs openblas         # 应打印 -L/usr/lib64 -lopenblas

# (3) 终极验证：从源码编译这几个包（CI 实际会做的事；scipy 是关键项）
pip install --no-binary=:all: 'scipy==1.18.0'
pip install --no-binary=:all: 'pillow==12.3.0'
pip install --no-binary=:all: 'outlines-core==0.1.26'
pip install --no-binary=:all: 'xgrammar==0.2.1'
```

第 (2) 步过了，scipy 基本就稳了；第 (3) 步 scipy 成功即整体合格。

## 6. 交付物

- 新镜像 tag，例如 `c-41:manylinux_2_38-loongarch64-2026081X`，推到
  `ext-artifactory.denglin.com:8082/ci-docker-images/c-41`。
- 把新 tag 告知 shaobo.xie，由我更新 `scripts/run/config_env.json` 里
  `docker_images.build.loongarch64` 与 `docker_images.runtime.loongarch64.openeuler` 两个字段，并跑一次 loong CI 验证。

## 7. 附录：依赖 → 工具链对应表

| Python 包（CI 实测版本） | 类型 | 需要的系统包 |
|---|---|---|
| outlines-core 0.1.26（经 outlines 0.1.11） | Rust 扩展 | rust, cargo |
| tiktoken | Rust 扩展 | rust, cargo |
| xgrammar 0.2.1 | C++/cmake | cmake, pkgconfig |
| torchao 0.9.0 | torch C++ 扩展 | gcc（镜像已有） |
| pillow 12.3.0 | C 扩展 | zlib/libjpeg-turbo/freetype/libtiff/libwebp-devel |
| scipy 1.18.0 | Fortran + BLAS | gcc-gfortran, openblas-devel **+ 第 4 节的 openblas.pc** |
| numpy 2.3.x | — | 已有 wheel / 正常 |
| av 18.0.0（PyAV） | C + FFmpeg | ❌ 仓库无 ffmpeg-devel，loong 上平台排除 |

---

**联系人**：shaobo.xie　　**验证用 CI job**：`compile_test_sglang_loongarch64`（分支 `dl-loongarch64-outlines-rust`）
