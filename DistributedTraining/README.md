# 说明
# 安装apex库
<font color="#dd0000">注：pip install apex的是其他同名库，需要从源码安装</font><br /> 
```shell
git clone https://github.com/NVIDIA/apex
cd apex
# 同时安装C++扩展
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# Apex 同样支持 Python-only build (required with Pytorch 0.4) via
pip install -v --no-cache-dir ./

```
<font color="#dd0000">注：若安装C++扩展，遇到报错
RuntimeError: Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries.  Pytorch binaries were compiled with Cuda 11.7.
</font><br /> 
```shell
# 查看当前环境使用的nvcc所有版本路径
whereis nvcc
# 找到需要的cuda11.7版本的nvcc，将其他的nvcc环境变量去除（可以暂时将nvcc改名为nvcc_bak,最后再改回来）
再执行安装C++扩展命令
# 
```
