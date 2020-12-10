module load conda/5.0.1-python3.6
source activate virt_pytorch

## Mount the squashfs image of Joconde on the Jocond folder
sudo mountimg /data/wimmics/user/abobashe/Joconde.squashfs /home/abobashe/Joconde

time python3 ./main.py train \
--image-root ~/Joconde/joconde \
--dataset '10_classes' \
--dataset-descr-file '~/Datasets/Joconde/Ten classes/dataset1.csv' \
--batch-size 64 \
--arch 'inception_v3' \
--finetuning \
--epochs 20 \
--param-file-suffix 2000.7 \
--optim Adam \
--activation sigmoid \
--loss BCE \
--use-weights \
--print-freq 5000 \
--workers 4
