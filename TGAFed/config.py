def set_config(args):

    args.gpu_mem = 7 # Gbyte (adjust this as needed)
    args.dataset_path = '/mnt/home/wym/TGAFed/data/dataset/table1'  # for datasets
    # args.dataset_path = '/mnt/home/wym/TGAFed/data/dataset/table2'  # for datasets
    # args.dataset_path = '/mnt/home/wym/TGAFed/data/dataset/table3'  # for datasets
    # args.dataset_path = '/mnt/home/wym/TGAFed/data/dataset/table5'  # for datasets
    # args.dataset_path = '/mnt/home/wym/FedMatch/data/dataset/'
    args.output_path = '/mnt/home/wym/TGAFed/data/outputs/'  # for logs, weights, etc.
    
    args.archi = 'resnet9'
    args.dataset_id_to_name = {0: 'cifar_10', 1: 'cifar_100', 2: 'svhn', 3: 'cinic_10'}

    
    # scenarios
    if 'lc' in args.task:
        args.scenario = 'labels-at-client'
        args.num_labels_per_class = 5
        if args.label_unlabel == 1:#1:10
            if 'c100' in args.task:  #56000/100/10=56
                args.num_labels_per_class = 5  #5+49
            elif 'svhn' in args.task: #75000/100/10=75
                args.num_labels_per_class = 7  #7+48
            elif 'c10' in args.task: #54000/100/10=54
                args.num_labels_per_class = 5  #5+51
            elif 'cinic_10' in args.task: #110000/100/10=110
                args.num_labels_per_class = 10  #10+100
        elif args.label_unlabel == 2: #2:9
            if 'c100' in args.task:
                args.num_labels_per_class = 10 #10+44
            elif 'svhn' in args.task:
                args.num_labels_per_class = 14 #14+61
            elif 'c10' in args.task:
                args.num_labels_per_class = 10 #10+46
            elif 'cinic_10' in args.task:
                args.num_labels_per_class = 20 #20+110
        elif args.label_unlabel == 3: #3:8
            if 'c100' in args.task:
                args.num_labels_per_class = 15 #15+39
            elif 'svhn' in args.task:
                args.num_labels_per_class = 21 #10+46
            elif 'c10' in args.task:
                args.num_labels_per_class = 15 #15+31
            elif 'cinic_10' in args.task:
                args.num_labels_per_class = 30 #30+80
        args.num_epochs_client = 1 
        args.batch_size_client = 10 # for labeled set
        args.num_epochs_server = 0
        args.batch_size_server = 0
        args.num_epochs_server_pretrain = 0
        args.lr_factor = 3
        args.lr_patience = 5
        args.lr_min = 1e-20

    # tasks
    if 'biid' in args.task or 'bimb'in args.task: 
        args.sync = False
        args.num_tasks = 1
        if 'c100' in args.task:
            if args.frac_clients == 0.2:
                args.num_clients = 10 
            elif args.frac_clients == 0.4:
                args.num_clients = 5
            args.num_rounds = 200    
        elif 'svhn' in args.task:
            if args.frac_clients == 0.05:
                args.num_clients = 100
            elif args.frac_clients == 0.07:
                args.num_clients = 75  
            elif args.frac_clients == 0.1:
                args.num_clients = 50  
            args.num_rounds = 300
        elif 'c10' in args.task:
            if args.frac_clients == 0.05:
                args.num_clients = 100
            elif args.frac_clients == 0.07:
                args.num_clients = 75  
            elif args.frac_clients == 0.1:
                args.num_clients = 50  
            args.num_rounds = 200
        elif 'cinic_10' in args.task:
            if args.frac_clients == 0.05:
                args.num_clients = 100
            elif args.frac_clients == 0.07:
                args.num_clients = 75  
            elif args.frac_clients == 0.1:
                args.num_clients = 50  
            args.num_rounds = 200   
    
    # datasets
    
    if 'c100' in args.task:  #60000
        args.dataset_id = 1
        args.num_classes = 100
        args.num_test = 4000  #        args.num_train = 54000
        args.num_valid = 2000
        args.batch_size_test = 100
    elif 'svhn' in args.task:
        args.dataset_id = 2
        args.num_classes = 10
        args.num_test = 14000   #        args.num_train = 75000   
        args.num_valid = 10289
        args.batch_size_test = 100    
    elif 'c10' in args.task:
        args.dataset_id = 0
        args.num_classes = 10
        args.num_test = 2000           #args.num_train = 56000
        args.num_valid = 2000
        args.batch_size_test = 100
    elif 'cinic_10' in args.task:
        args.dataset_id = 3
        args.num_classes = 10
        args.num_test = 80000           #args.num_train = 110000
        args.num_valid = 80000
        args.batch_size_test = 400

    # base networks
    if args.archi in ['resnet9']:
        args.lr = 1e-3
        args.wd = 1e-4

    # hyper-parameters
    if args.model in ['fedmatch']:
        args.num_helpers = 2
        args.confidence = 0.85   #置信度
        args.psi_factor = 0.2   #调整权重的缩放因子
        args.h_interval = 10   #指定在每个多少轮次之后进行一次帮助者模型的选择

        args.lambda_s = 10 # supervised learning
        args.lambda_i = 1e-2 # inter-client consistency
        args.lambda_a = 1e-2 # agreement-based pseudo labeling
        if 'c100' in args.task:
            args.lambda_l2 = 10  #c100:10
            args.lambda_l1 = 1   #c100:1
        elif 'svhn' in args.task:
            args.lambda_l2 = 10  #svhn:10
            args.lambda_l1 = 1e-5   #svhn:1e-5          
        elif 'c10' in args.task:
            args.lambda_l2 = 10  #c10:10
            args.lambda_l1 = 1e-4   #c10:1e-4
        elif 'cinic_10' in args.task:
            if 'biid' in args.task:
                args.lambda_l2 = 10  #cinic_10:10
                args.lambda_l1 = 1e-4   #cinic_10:1e-4      
            else:
                args.lambda_l2 = 10  #cinic_10:10
                args.lambda_l1 = 1e-5   #cinic_10:1e-4                      
        args.l1_thres = 1e-6 * 5
        args.delta_thres = 1e-5 * 5

        #TGAFed参数
        args.lambda_c = 1e-1  #exp_loss        
    elif args.model in ['fedavg']:
        args.psi_factor = 0.2   #调整权重的缩放因子

        args.lambda_u = 1e-2
        args.lambda_s = 10 # supervised learning

        args.lambda_l1 = 1e-4
        args.l1_thres = 1e-6 * 5
        args.delta_thres = 1e-5 * 5
    elif args.model in ['fedprox']:
        args.psi_factor = 0.2   #调整权重的缩放因子
        args.lambda_u = 1e-2
        args.lambda_s = 10 # supervised learning
        args.fedprox_mu = 0.1

        args.lambda_l1 = 1e-4
        args.l1_thres = 1e-6 * 5
        args.delta_thres = 1e-5 * 5    

    return args


