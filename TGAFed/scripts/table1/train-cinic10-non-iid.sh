python3 main.py --work-type gen_data --task lc-bimb-cinic_10 --label_unlabel 1 --frac-client 0.05

######################  实验1  #########################
##FedAvg-UDA
python3 main.py  --gpu 2 --work-type train --model fedavg --frac-client 0.05 --task lc-bimb-cinic_10 --semi uda --label_unlabel 1 

##FedProx-UDA
python3 main.py  --gpu 2 --work-type train --model fedprox --frac-client 0.05 --task lc-bimb-cinic_10 --semi uda --label_unlabel 1

##FedAvg-FixMatch
python3 main.py  --gpu 2 --work-type train --model fedavg --frac-client 0.05 --task lc-bimb-cinic_10 --semi fixmatch --label_unlabel 1

##FedProx-FixMatch
python3 main.py  --gpu 2 --work-type train --model fedprox --frac-client 0.05 --task lc-bimb-cinic_10 --semi fixmatch --label_unlabel 1

##FedMatch
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-cinic_10 --label_unlabel 1

##TGAFed
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-cinic_10 --label_unlabel 1 --aggregate True


