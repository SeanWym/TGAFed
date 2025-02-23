#c10  noniid
python3 main.py --work-type gen_data --task lc-bimb-c10  --label_unlabel 1 --frac-client 0.05   ##原始
python3 main.py --work-type gen_data --task lc-bimb-c10-07  --label_unlabel 1 --frac-client 0.07
python3 main.py --work-type gen_data --task lc-bimb-c10-10  --label_unlabel 1 --frac-client 0.1

#c100  noniid
python3 main.py --work-type gen_data --task lc-bimb-c100-1  --label_unlabel 1 --frac-client 0.2
python3 main.py --work-type gen_data --task lc-bimb-c100-40  --label_unlabel 1 --frac-client 0.4

#svhn  noniid
python3 main.py --work-type gen_data --task lc-bimb-svhn-1  --label_unlabel 1 --frac-client 0.05
python3 main.py --work-type gen_data --task lc-bimb-svhn-07  --label_unlabel 1 --frac-client 0.07
python3 main.py --work-type gen_data --task lc-bimb-svhn-10  --label_unlabel 1 --frac-client 0.1

#cinic_10  noniid
python3 main.py --work-type gen_data --task lc-bimb-cinic_10  --label_unlabel 1 --frac-client 0.05   ##原始
python3 main.py --work-type gen_data --task lc-bimb-cinic_10-07  --label_unlabel 1 --frac-client 0.07
python3 main.py --work-type gen_data --task lc-bimb-cinic_10-10  --label_unlabel 1 --frac-client 0.1

######################  实验3  #########################  针对不同客户端的数量进行实验
##########################FedMatch
#c10  noniid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10 --label_unlabel 1 #原始
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.07 --task lc-bimb-c10-07 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.1 --task lc-bimb-c10-10 --label_unlabel 1

#c100  noniid
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.4 --task lc-bimb-c100-40  --label_unlabel 1

#svhn  noniid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-svhn-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.07 --task lc-bimb-svhn-07  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.1 --task lc-bimb-svhn-10  --label_unlabel 1

#cinic_10  noniid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-cinic_10 --label_unlabel 1 #原始
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.07 --task lc-bimb-cinic_10-07 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.1 --task lc-bimb-cinic_10-10 --label_unlabel 1

##########################Fedavg-FixMatch
#c10  noniid
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-c10 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.07 --task lc-bimb-c10-07 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.1 --task lc-bimb-c10-10 --label_unlabel 1

#c100  noniid
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.2 --task lc-bimb-c100-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.4 --task lc-bimb-c100-40  --label_unlabel 1

#svhn  noniid
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-svhn-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.07 --task lc-bimb-svhn-07  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.1 --task lc-bimb-svhn-10  --label_unlabel 1

#cinic_10  noniid
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-cinic_10 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.07 --task lc-bimb-cinic_10-07 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.1 --task lc-bimb-cinic_10-10 --label_unlabel 1

