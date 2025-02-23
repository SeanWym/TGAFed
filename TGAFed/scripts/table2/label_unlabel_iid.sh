
################################iid

#c10  iid
python3 main.py --work-type gen_data --task lc-biid-c10  --label_unlabel 1 --frac-client 0.05   ##原始
python3 main.py --work-type gen_data --task lc-biid-c10-2  --label_unlabel 2 --frac-client 0.05
python3 main.py --work-type gen_data --task lc-biid-c10-3  --label_unlabel 3 --frac-client 0.05

#c100  iid
python3 main.py --work-type gen_data --task lc-biid-c100-1  --label_unlabel 1 --frac-client 0.2
python3 main.py --work-type gen_data --task lc-biid-c100-2  --label_unlabel 2 --frac-client 0.2
python3 main.py --work-type gen_data --task lc-biid-c100-3  --label_unlabel 3 --frac-client 0.2

#svhn  iid
python3 main.py --work-type gen_data --task lc-biid-svhn-1  --label_unlabel 1 --frac-client 0.05
python3 main.py --work-type gen_data --task lc-biid-svhn-2  --label_unlabel 2 --frac-client 0.05
python3 main.py --work-type gen_data --task lc-biid-svhn-3  --label_unlabel 3 --frac-client 0.05

#cinic_10  iid
python3 main.py --work-type gen_data --task lc-biid-cinic_10  --label_unlabel 1 --frac-client 0.05   ##原始
python3 main.py --work-type gen_data --task lc-biid-cinic_10-2  --label_unlabel 2 --frac-client 0.05 
python3 main.py --work-type gen_data --task lc-biid-cinic_10-3  --label_unlabel 3 --frac-client 0.05 

######################  实验2  #########################  针对 标记数据：未标记数据  比例
##########################FedMatch
#c10  iid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10 --label_unlabel 1  #原始
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-2 --label_unlabel 2
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-c10-3 --label_unlabel 3

#c100  iid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-c100-1  --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-c100-2  --label_unlabel 2
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.2 --task lc-biid-c100-3  --label_unlabel 3

#svhn  iid
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-svhn-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-svhn-2  --label_unlabel 2
python3 main.py  --gpu 1 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-svhn-3  --label_unlabel 3

#cinic_10  iid
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-cinic_10 --label_unlabel 1  #原始
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-cinic_10-2 --label_unlabel 2 --aggregate True
python3 main.py  --gpu 2 --work-type train --model fedmatch --frac-client 0.05 --task lc-biid-cinic_10-3 --label_unlabel 3 --aggregate True

##########################Fedavg-FixMatch
#c10  iid
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-c10-1 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-c10-2 --label_unlabel 2
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-c10-3 --label_unlabel 3

#c100  iid
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.2 --task lc-biid-c100-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.2 --task lc-biid-c100-2  --label_unlabel 2
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.2 --task lc-biid-c100-3  --label_unlabel 3

#svhn  iid
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-svhn-1  --label_unlabel 1
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-svhn-2  --label_unlabel 2
python3 main.py  --gpu 1 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-svhn-3  --label_unlabel 3

#cinic_10  iid
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-cinic_10 --label_unlabel 1
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-cinic_10-2 --label_unlabel 2
python3 main.py  --gpu 2 --work-type train --model fedavg --semi fixmatch --frac-client 0.05 --task lc-biid-cinic_10-3 --label_unlabel 3
