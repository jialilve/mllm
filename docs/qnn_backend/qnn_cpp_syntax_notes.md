# C++ 语法速查（围绕 pipeline 代码）

> 当需要快速理解现有代码结构但又不想系统学习 C++ 时，可把这里当作“目录”。更深入内容可查《C++ Primer》。

## 1. `std::function` 与 Lambda
- **Lambda**：`[capture](params) -> return { body }`。`[&]` 表示按引用捕获当前作用域变量（例：`executeFunc = [&](int chunk_id, int graphIdx) { ... };`）。
- **`std::function<void(int,int)>`**：类型擦除后的可调用对象。可接收函数、lambda、仿函数。  
  - `std::function<void(int,int)> f = [](int a,int b){...};`
  - 可以像普通函数一样调用：`f(x,y);`

## 2. `auto` 与 `decltype`
- `auto var = expression;` 让编译器推导类型，常见于迭代器、lambda。  
- `decltype(expr)` 返回表达式类型，多用于 `using foo = decltype(bar);`。

## 3. 智能指针
- `std::shared_ptr<T>`：引用计数，适合共享所有权。
- `std::make_shared<T>(args...)`：更安全高效的创建方式。
- 访问底层对象用 `ptr->method()`。

## 4. STL 容器基础
- `std::vector<T>`：动态数组，常见操作：
  - `vec.push_back(value);`
  - `vec[i]` / `vec.at(i)` 访问元素。
  - `vec.size()` 获取长度。
- `std::queue<T>`：先进先出，`push`, `pop`, `front`.

## 5. `std::array` / 初始化列表
- `std::array<int,2> arr = {1,2};` 固定大小。
- 初始化列表 `{...}` 可直接赋给 vector/array。

## 6. 并发相关关键字
- `#pragma omp parallel for num_threads(N)`：OpenMP 指令，创建 N 个线程执行 for 循环。
- `std::future<T>` / `std::async`：标准异步任务：
  ```cpp
  auto fut = std::async(std::launch::async, func, args...);
  T result = fut.get();   // 阻塞直到 func 返回
  ```

## 7. 命名空间
- `namespace mllm { ... }`：逻辑分组，调用时需 `mllm::Class`.
- `using namespace` 会把命名空间导入当前作用域（不建议在头文件中使用）。

## 8. 头文件/源文件结构
- `.hpp` 放声明，`.cpp` 放实现。  
- 常见布局：
  ```cpp
  class Foo {
  public:
      Foo();
      void bar(int v);
  private:
      int value_;
  };
  ```
  ```cpp
  Foo::Foo() : value_(0) {}
  void Foo::bar(int v) { value_ = v; }
  ```

## 9. `struct` 与 `class`
- 默认可见性不同：`struct` 默认 public，`class` 默认 private。  
- 其他语义一致。

## 10. `std::pair` / `std::tuple`
- `std::pair<A,B>`：两个元素，成员 `.first`, `.second`。  
- C++17 起可以结构化绑定：`auto [x,y] = pair_obj;`

## 11. 断言与日志
- `assert(condition);` 调试期间检查。  
- 项目里常见 `MLLM_INFO`, `MLLM_ERROR` 等宏，类似 `printf`。

## 12. `constexpr` / `const`
- `const`：不可修改变量。  
- `constexpr`：编译期常量，可用于数组大小等。

## 13. 范型 & 模板
- 声明：`template<typename T> T add(T a, T b) { return a + b; }`
- 使用：`add<int>(1,2)` 或 `add(1,2)`（自动推导）。

> 以上涵盖 pipeline 相关代码最常出现的语法。遇到陌生写法时，可先在这里定位主题，再去《C++ Primer》查阅细节。


