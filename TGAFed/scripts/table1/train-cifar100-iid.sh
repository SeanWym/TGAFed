######################  实验1  #########################
##FedAvg-UDA
python3 main.py  --gpu 2 --work-type train --model fedavg --frac-client 0.2 --task lc-biid-c100-1 --semi uda --label_unlabel 1

##FedProx-UDA
python3 main.py  --gpu 2 --work-type train --model fedprox --frac-client 0.2 --task lc-biid-c100-1 --semi uda --label_unlabel 1

##FedAvg-FixMatch
python3 main.py  --gpu 2 --work-type train --model fedavg --frac-client 0.2 --task lc-biid-c100-1 --semi fixmatch --label_unlabel 1

##FedProx-FixMatch
python3 main.py  --gpu 2 --work-type train --model fedprox --frac-client 0.2 --task lc-biid-c100-1 --semi fixmatch --label_unlabel 1

##FedMatch
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-c100-1 --label_unlabel 1

##TGAFed
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-c100-1 --label_unlabel 1 --aggregate True
