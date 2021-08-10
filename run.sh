#!/usr/bin/bash
set -x

./build/src/vgg16_circuit_run vgg1 vgg1.ins vgg1.wit vgg1.rel nchw
./build/src/vgg16_circuit_run vgg2 vgg2.ins vgg2.wit vgg2.rel nchw
./build/src/vgg16_circuit_run vgg3 vgg3.ins vgg3.wit vgg3.rel nchw
./build/src/vgg16_circuit_run vgg4 vgg4.ins vgg4.wit vgg4.rel nchw
./build/src/vgg16_circuit_run vgg5 vgg5.ins vgg5.wit vgg5.rel nchw
./build/src/vgg16_circuit_run vgg11 vgg11.ins vgg11.wit vgg11.rel nchw
./build/src/vgg16_circuit_run vgg13 vgg13.ins vgg13.wit vgg13.rel nchw
./build/src/vgg16_circuit_run vgg16 vgg16.ins vgg16.wit vgg16.rel nchw