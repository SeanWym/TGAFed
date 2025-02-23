#c10  noniid
python3 main.py --work-type gen_data --task lc-bimb-c10-dir0.8  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.8   
python3 main.py --work-type gen_data --task lc-bimb-c10-dir0.5  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.5  
python3 main.py --work-type gen_data --task lc-bimb-c10-dir0.2  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.2  

#c100  noniid
python3 main.py --work-type gen_data --task lc-bimb-c100-1  --label_unlabel 1 --frac-client 0.2 --dirichlet 0.8 #原始使用的就是0.8 可以直接使用
python3 main.py --work-type gen_data --task lc-bimb-c100-dir0.5  --label_unlabel 1 --frac-client 0.2 --dirichlet 0.5
python3 main.py --work-type gen_data --task lc-bimb-c100-dir0.2  --label_unlabel 1 --frac-client 0.2 --dirichlet 0.2

#svhn  noniid
python3 main.py --work-type gen_data --task lc-bimb-svhn-1  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.8 #原始使用的就是0.8 可以直接使用
python3 main.py --work-type gen_data --task lc-bimb-svhn-dir0.5  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.5
python3 main.py --work-type gen_data --task lc-bimb-svhn-dir0.2  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.2

#cinic_10  noniid
python3 main.py --work-type gen_data --task lc-bimb-cinic_10  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.8   ##原始
python3 main.py --work-type gen_data --task lc-bimb-cinic_10-dir0.5  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.5  
python3 main.py --work-type gen_data --task lc-bimb-cinic_10-dir0.2  --label_unlabel 1 --frac-client 0.05 --dirichlet 0.2  

######################  实验3  #########################  针对不同客户端的数量进行实验
##########################FedMatch
#c10  noniid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-dir0.8 --label_unlabel 1 #原始
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-dir0.5 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-c10-dir0.2 --label_unlabel 1

#c100  noniid
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-dir0.5  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.2 --task lc-bimb-c100-dir0.2  --label_unlabel 1

#svhn  noniid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-svhn-dir0.5  --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-svhn-dir0.2  --label_unlabel 1

#cinic_10  noniid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-cinic_10 --label_unlabel 1 #原始
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-cinic_10-dir0.5 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-bimb-cinic_10-dir0.2 --label_unlabel 1

##########################Fedavg-FixMatch
#c10  noniid
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-c10-dir0.8 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-c10-dir0.5 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-c10-dir0.2 --label_unlabel 1

#c100  noniid
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.2 --task lc-bimb-c100-dir0.5  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.2 --task lc-bimb-c100-dir0.2  --label_unlabel 1

#svhn  noniid
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-svhn-dir0.5  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-svhn-dir0.2  --label_unlabel 1

#c10  noniid
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-cinic_10 --label_unlabel 1 #原始
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-cinic_10-dir0.5 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-bimb-cinic_10-dir0.2 --label_unlabel 1


