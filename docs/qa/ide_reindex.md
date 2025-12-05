## IDE 索引重载速查

当手动生成或更新 `compile_commands.json`（例如新建 `build-android-qnn-dbg` 并在仓库根目录创建符号链接）后，必须让 IDE/语言服务器重新读取该编译数据库，否则 `Ctrl+点击` 等跳转仍会使用旧索引。下面列出常用编辑器和插件的快速操作。

### Cursor / VS Code
- `Ctrl + Shift + P` 打开命令面板，执行 `Reload Window`（最可靠，相当于快速重启 IDE）。
- 如果安装官方 C/C++ 扩展，可执行 `C/C++: Reset IntelliSense database`，无需重启整个 IDE。
- 也可执行 `Developer: Restart Extension Host`，在不关闭窗口的情况下重新加载所有扩展。

### clangd 扩展
- `Ctrl + Shift + P` → `Clangd: Restart language server`，触发 clangd 重新读取 `compile_commands.json`。
- 如果命令面板里看不到，先输入 `>Clangd` 过滤即可。

### CMake Tools
- 右下角 `CMake` 状态栏按钮 → `Reload/Configure` 或 `CMake: Configure`。
- 命令面板执行 `CMake: Scan for Kits` 也会强制刷新配置并重新生成数据库。

### 通用排查步骤
- 确认语言服务器配置中的 `compileCommands` 指向 `/root/mllm_v2/compile_commands.json`（符号链接也可以）。
- 若仍无效，可禁用再启用相关扩展，或最后再尝试关闭重开 IDE。
- 核对 `compile_commands.json` 的时间戳是否更新，以及路径中是否包含目标文件（如 `examples/qwen_npu/main.cpp`）。

按照以上任一天操作执行完，IDE 会重新索引项目，`model.trace` 等符号跳转即可恢复正常。


