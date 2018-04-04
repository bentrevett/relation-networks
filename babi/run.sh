#!/bin/bash

cd data

wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-1.tar.gz

dtrx -one here tasks_1-20_v1-1.tar.gz

cd ..

python babi-generator.py