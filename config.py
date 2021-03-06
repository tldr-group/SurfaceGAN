import json


class Config():
    """Config class
    """
    def __init__(self, tag):
        self.tag = tag
        self.path = f'runs/{self.tag}'
        self.data_path = 'data/2D surface profiles/2D surface profiles/Vertical/pp_2_'
        self.net_type = 'upsizegan'
        self.l = 128
        self.n_phases = 2
        self.img_channels = 2
        # Training hyperparams
        self.batch_size = 32
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.num_epochs = 250
        self.iters = 1000
        self.lrg = 0.001
        self.lr = 0.001
        self.Lambda = 10
        self.critic_iters = 5
        self.lz = 4
        self.lf = 4
        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'
        self.nz = 32
        # Architecture
        self.lays = 5
        self.laysd = 6
        # kernel sizes
        self.dk, self.gk = [4]*self.laysd, [4]*self.lays
        self.ds, self.gs = [2]*self.laysd, [2]*self.lays
        self.df, self.gf = [self.n_phases, 32, 32, 32, 32, 32, 1], [
            self.nz, 512, 256, 256, 128, self.img_channels]
        self.dp = [1, 1, 1, 1, 1, 0]
        self.gp = [1, 1, 1, 1, 1, 1]

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(j, f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_net_params(self):
        return self.dk, self.ds, self.df, self.dp, self.gk, self.gs, self.gf, self.gp
    
    def get_train_params(self):
        return self.l, self.batch_size, self.beta1, self.beta2, self.num_epochs, self.iters, self.lrg, self.lr, self.Lambda, self.critic_iters, self.lz, self.nz


