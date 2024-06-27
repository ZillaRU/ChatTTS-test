cd ChatTTS/model
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
cp llama.cpy* ..
cd ../../..
export LD_LIBRARY_PATH=./LLM-TPU/support/lib_pcie # export LD_LIBRARY_PATH=./LLM-TPU/support/lib_soc
ldd ChatTTS/model/llama.cpython-310-x86_64-linux-gnu.so
gdb --args python test_main.py

docker start 90d4d279eda3  cpu版本 对比
docker start rzy0523 转bmodel环境 trace环境 transformers 4.32 替换掉modeling_llama.py
docker start chattts 编cpp module，测试tpu版chattts，5号映射到device0
