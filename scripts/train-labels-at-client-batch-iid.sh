python3 ../main.py  --gpu 0,1,2,3,4 \
					--work-type train \
					--model fedmatch \
					--frac-client 0.05 \
					--task lc-biid-c10 \
##################cifar10
###iid  200轮
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10 
###noniid  200轮
python3 main.py  --gpu 5 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10


##################cifar100
###iid  200
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-c100
###noniid  200轮
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100

##################SVHN
###iid  200
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-SVHN-1 --svhn_unlabels_bili 1
###noniid  200轮
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-SVHN-1 --svhn_unlabels_bili 1



#c10  TGAFed iid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-9
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-8
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-7
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-6
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-5
# c10  FedMatch  iid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-9
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-8
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-7
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-6
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-5


##########################
#c10  TGAFed non-iid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-9
python3 main.py  --gpu 6 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-8
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-7
python3 main.py  --gpu 3 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-6
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-5
# c10  FedMatch  non-iid
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-9
python3 main.py  --gpu 7 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-8
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-7
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-6
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-5


### c100 noniid  200轮
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-8
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-6
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-4
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-2