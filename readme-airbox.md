安装依赖
查看pybind11路径`pip3 show pybind11`，将
设置环境变量
下载bmodel
`cmake -DCMAKE_HOST_SYSTEM_PROCESSOR=aarch64 ..`(for SoC mode)
`cmake ..`(for PCIE mode)
`export LD_LIBRARY_PATH=./LLM-TPU/support/lib-soc`