"""
Training of my_gan
"""

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import sklearn.svm as svm
from sklearn.utils import column_or_1d
from sklearn.manifold import TSNE


seed = 2
torch.manual_seed(seed)
np.random.seed(seed)


def svc(kernel):
    return svm.SVC(kernel=kernel, decision_function_shape="ovo")

def nusvc():
    return svm.NuSVC(decision_function_shape="ovo")

def linearsvc():
    return svm.LinearSVC(multi_class="ovr")

def modelist():
    modelist = []
    kernalist = {"linear", "poly", "rbf", "sigmoid"}
    for each in kernalist:
        modelist.append(svc(each))
    modelist.append(nusvc())
    modelist.append(linearsvc())
    return modelist


def svc_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    acu_train = model.score(x_train, y_train)
    acu_test = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    recall = recall_score(y_test, y_pred, average="macro")
    return acu_train, acu_test, recall

## Dataset
class Transform_indicators(Dataset):
    def __init__(self, is_train_set=True):
        super(Transform_indicators, self).__init__()
        self.filename = r"E:\西南大学\GAN\SFRA的指标程序\transformer_indicators_norm_data.xlsx"
        # self.filename = r"E:\陈宇文件\GAN\程序\SFRA的指标程序\transformer_indicators_norm_data.xlsx"

        data = np.array(pd.read_excel(self.filename, sheet_name='Sheet1'))
        self.SFRA_indicator = torch.tensor(data[:, 0:30]).float()
        self.SFRA_label = torch.tensor(data[:, 30]).long()


    def __getitem__(self, index):
        # batch_size * 1 * 71
        input = self.SFRA_indicator[index,:].view(1, 30)
        # batch_size * 1
        label = self.SFRA_label[index].view(1)
        return input, label

    def __len__(self):
        return self.SFRA_indicator.shape[0]

# hyper parameters
batch_size = 6
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
SEQ_SIZE = 30
CHANNELS_SEQ = 1
NUM_CLASSES = 3
GEN_EMBEDDING = 10
Z_DIM = 100
NUM_EPOCHS = 10000
FEATURES_CRITIC = 5
FEATURES_GEN = 1
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10




SFRA_Data = Transform_indicators()
# torch.Size([6, 1, 30]) torch.Size([6, 1])
SFRA_DataLoader = DataLoader(SFRA_Data, batch_size=batch_size, shuffle=True)


# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_SEQ, FEATURES_GEN, NUM_CLASSES, SEQ_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_SEQ, FEATURES_CRITIC, NUM_CLASSES, SEQ_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))



# for tensorboard plotting
number = 100
fixed_noise = torch.randn(100, Z_DIM, 1).to(device)
train_noise = torch.randn(1000, Z_DIM, 1).to(device)
# plot label
plot_label_0 = torch.tensor([0]).repeat(100, 1).to(device)
plot_label_1 = torch.tensor([1]).repeat(100, 1).to(device)
plot_label_2 = torch.tensor([2]).repeat(100, 1).to(device)

# t-sne
tsne = TSNE(n_components=2, init='pca', random_state=0)

# train svm label
train_label_0 = torch.tensor([0]).repeat(1000, 1).to(device)
train_label_1 = torch.tensor([1]).repeat(1000, 1).to(device)
train_label_2 = torch.tensor([2]).repeat(1000, 1).to(device)



writer_figure = SummaryWriter(f"logs/GAN_transformer_SFRA/figure")
writer_curve = SummaryWriter(f"logs/GAN_transformer_SFRA/curve")
writer_svm_curve = SummaryWriter(f"logs/GAN_transformer_SFRA/svm_curve")

step = 0

# real data

# filename = r"E:\陈宇文件\GAN\程序\SFRA的指标程序\transformer_indicators_norm_data.xlsx"
filename = r"E:\西南大学\GAN\SFRA的指标程序\transformer_indicators_norm_data.xlsx"
data_frama = pd.read_excel(filename, sheet_name='Sheet1')
data_test = data_frama.iloc[:,:30]
label_test = data_frama.iloc[:,30]
real_data = np.array(data_frama)

gen.train()
critic.train()
best_gen_loss = 0


for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, labels) in enumerate(SFRA_DataLoader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 8 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(SFRA_DataLoader)} \
                          Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            # take out (up to) 32 examples
            writer_curve.add_scalar("loss_gen", loss_gen, global_step=step)
            if loss_gen < best_gen_loss:
                torch.save(gen.state_dict(), rf"E:\陈宇文件\GAN\程序\gen_best\gen{loss_gen}.pt")
                best_gen_loss = loss_gen

            writer_curve.add_scalar("loss_critic", loss_critic, global_step=step)
            step += 1




    filename = r"E:\西南大学\GAN\SFRA的指标程序\transformer_indicators_norm_data.xlsx"
    # filename = r"E:\陈宇文件\GAN\程序\SFRA的指标程序\transformer_indicators_norm_data.xlsx"
    data = np.array(pd.read_excel(filename, sheet_name='Sheet1'))
    with torch.no_grad():
        if (epoch+1) % 100 == 0:
            plot_fake_0 = gen(fixed_noise, plot_label_0)
            plot_fake_1 = gen(fixed_noise, plot_label_1)
            plot_fake_2 = gen(fixed_noise, plot_label_2)
            # plot scatter
            SFRA_indicator = real_data[:, 0:30].reshape(-1, 30)
            SFRA_label = real_data[:, 30].reshape(-1)
            # fake = fake.cpu().numpy().reshape(-1, 30)
            # gen_label = gen_label.cpu().numpy().reshape(-1)
            plot_fake_0 = plot_fake_0.cpu().numpy().reshape(-1, 30)
            plot_fake_1 = plot_fake_1.cpu().numpy().reshape(-1, 30)
            plot_fake_2 = plot_fake_2.cpu().numpy().reshape(-1, 30)
            # plot  t-sne
            data_all = np.vstack((plot_fake_0, plot_fake_1, plot_fake_2, SFRA_indicator))
            result = tsne.fit_transform(data_all)
            fig_tsne = plt.figure()
            plt.scatter(result[0:number, 0], result[0:number, 1], c="r", marker="*")
            plt.scatter(result[number:number * 2, 0], result[number:number * 2, 1], c="r", marker="o")
            plt.scatter(result[number * 2:number * 3, 0], result[number * 2:number * 3, 1], c="r", marker="^")
            plt.scatter(result[number * 3:number * 3 + 21, 0], result[number * 3:number * 3 + 21, 1], c="g", marker='*')
            plt.scatter(result[number * 3 + 21:number * 3 + 36, 0], result[number * 3 + 21:number * 3 + 36, 1], c="g",
                        marker="o")
            plt.scatter(result[number * 3 + 36:number * 3 + 53, 0], result[number * 3 + 36:number * 3 + 53, 1], c="g",
                        marker="^")
            writer_figure.add_figure(f"t-sne", figure=fig_tsne, global_step=(epoch + 1))


        # save model and train svm
        if (epoch+1) % 1 == 0:
            # train svm
            # data
            train_fake_0 = gen(train_noise, train_label_0).cpu().detach().numpy().reshape(-1, 30)
            train_fake_1 = gen(train_noise, train_label_1).cpu().detach().numpy().reshape(-1, 30)
            train_fake_2 = gen(train_noise, train_label_2).cpu().detach().numpy().reshape(-1, 30)
            label_train = np.vstack((train_label_0.cpu().detach().numpy(), train_label_1.cpu().detach().numpy(), train_label_2.cpu().detach().numpy()))
            data_train = np.vstack((train_fake_0, train_fake_1, train_fake_2))
            label_train = label_train.ravel()
            label_test = label_test.ravel()
            model_num = 0
            for model in modelist():
                acu_train, acu_test, recall = svc_model(model, data_train, label_train, data_test, label_test)
                writer_svm_curve.add_scalar(f"svm_acu_test_{model_num}", acu_test, global_step=epoch + 1)
                writer_svm_curve.add_scalar(f"svm_acu_train_{model_num}",acu_train, global_step=epoch + 1)
                writer_svm_curve.add_scalar(f"svm_recall_{model_num}", recall, global_step=epoch + 1)
                model_num += 1
            torch.save(gen.state_dict(), rf"E:\西南大学\GAN\youtube资源\GANs\my_gan\model\gen{epoch}.pt")


            # torch.save(critic.state_dict(),rf"E:\陈宇文件\GAN\程序\my_gan\model\critic{epoch}.pt")


writer_svm_curve.close()
writer_figure.close()
writer_curve.close()





