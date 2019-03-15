Train model:
Run the bash file by command: ./train.sh
If you want to change training iterations or dataset or something else, please modify train.sh file. Details are given in that file.

Test model:
Run the bash file by command: ./val.sh

run demo.py:
first, please put your images and annotations in data/demo, such as:
data/demo/SET-xx/infrared
data/demo/SET-xx/visible
data/demo/SET-xx/xml

at the same time, please copy these files into data/EELABdevkit/data/images/infrared, data/EELABdevkit/data/images/visible, data/EELABdevkit/data/annotations

then, you can run demo.py by command: python3 ./tools/demo.py --method 'xxx' --testset 'xxx'
and all the results will be saved at 'data/demo\_result'
