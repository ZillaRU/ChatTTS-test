cd ChatTTS/model
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
cp llama.cpy* ..
cd ../../..
export LD_LIBRARY_PATH=./LLM-TPU/support/lib_pcie # export LD_LIBRARY_PATH=./LLM-TPU/support/lib_soc
ldd ChatTTS/model/llama.cpython-310-x86_64-linux-gnu.so
gdb --args python test_main.py