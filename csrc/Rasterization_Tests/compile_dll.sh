echo "Compiling"
g++ -c -fPIC -std=c++11 rasterizer.cpp -o rasterizer.o
echo "Generating shared library"
g++ -shared -Wl,-soname,libraster.so -o libraster.so rasterizer.o
