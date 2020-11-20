#encoding:utf-8

from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *
from module.base_module import *

#データセットAの、各データへのパスのフォーマット　make_datapath_listへの引数
dataroot_A = './dataset/group_A/**/*'
#データセットBの、各データへのパスのフォーマット
dataroot_B = './dataset/group_B/**/*'
#バッチサイズ
batch_size = 1
#エポック数
num_epochs = 1110
#generator,discriminatorのoptimizerに使う学習率
learning_rate = 0.0001
#Adamのweight decay(重み減衰)の度合い
weight_decay = 0.0001
#output_progress_intervalエポックごとに学習状況の画像を出力する
output_progress_interval = 1

#訓練データAの読み込み、データセット作成
path_list_A = make_datapath_list(dataroot_A)
transform_A = ImageModification(resize_pixel=256,x_move=[-0.05,0.05],y_move=[-0.05,0.05],min_scale=0.9)
train_dataset_A = GAN_Img_Dataset(file_list=path_list_A,transform=transform_A)
dataloader_A = torch.utils.data.DataLoader(train_dataset_A,batch_size=batch_size,shuffle=True)

#訓練データBの読み込み、データセット作成
path_list_B = make_datapath_list(dataroot_B)
transform_B = ImageModification(resize_pixel=256,x_move=[-0.1,0.1],y_move=[-0.1,0.25],min_scale=0.7)
train_dataset_B = GAN_Img_Dataset(file_list=path_list_B,transform=transform_B)
dataloader_B = torch.utils.data.DataLoader(train_dataset_B,batch_size=batch_size,shuffle=True)

#GPUが使用可能かどうか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

# #ネットワークを初期化するための関数
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		#平均0.0,標準偏差0.02となるように初期化
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('ConvTranspose2d') != -1:
		#平均0.0,標準偏差0.02となるように初期化
		nn.init.normal_(m.weight.data, 0.0, 0.02)

#各ネットワークのインスタンスを生成、デバイスに移動
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_GA = Discriminator(n_layers=7).to(device)
netD_GB = Discriminator(n_layers=7).to(device)
netD_LA = Discriminator(n_layers=5).to(device)
netD_LB = Discriminator(n_layers=5).to(device)
#ネットワークの初期化
netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_GA.apply(weights_init)
netD_GB.apply(weights_init)
netD_LA.apply(weights_init)
netD_LB.apply(weights_init)

#損失関数の初期化
L1_loss = nn.L1Loss().to(device)
MSE_loss = nn.MSELoss().to(device)
BCE_loss = nn.BCEWithLogitsLoss().to(device)
#Adam optimizersをGeneratorとDiscriminatorに適用
beta1 = 0.5
beta2 = 0.999
optimizerG = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),lr=learning_rate,betas=(beta1,beta2),weight_decay=weight_decay)
optimizerD = torch.optim.Adam(itertools.chain(netD_GA.parameters(),netD_GB.parameters(),netD_LA.parameters(),netD_LB.parameters()),lr=learning_rate,betas=(beta1,beta2),weight_decay=weight_decay)

#イテレーションを全部で何回実行することになるかを計算
iteration_per_epoch = len(path_list_A) if len(path_list_A)<len(path_list_B) else len(path_list_B)
total_iteration = num_epochs*iteration_per_epoch

#学習過程を追うための変数
G_losses = []
D_losses = []
iteration = 0

#学習過程を追うためのサンプルの作成
sample_image_num = 5#何枚のサンプルを用意するか
#訓練データAとBからサンプルを作成
def make_sample(path_regex,sample_img_num):
	path_list = make_datapath_list(path_regex)
	dataset = GAN_Img_Dataset(file_list=path_list,transform=ImageTransform(256))
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=sample_img_num,shuffle=False)
	return next(iter(dataloader))
#学習過程を追うためのサンプルを作成し変数に格納
sample_real_A = make_sample(dataroot_A,sample_image_num).to(device)
sample_real_B = make_sample(dataroot_B,sample_image_num).to(device)

#学習過程を追うための、画像出力用関数
def output_how_much_progress(filename,imgs,normalize=True):
	#引数はfilename以外はいずれもtorch.Size([sample_image_num,3,256,256])
	output_imgs = []
	for im in imgs:
		output_imgs.append(torchvision.utils.make_grid(im,nrow=sample_image_num,padding=10))
	output_imgs = torch.stack(output_imgs,dim=0)
	output_imgs = torchvision.utils.make_grid(output_imgs,nrow=1,padding=100)
	vutils.save_image(output_imgs,filename,normalize=normalize)

#学習開始
print("Starting Training")

#学習開始時刻を保存
t_epoch_start = time.time()
#エポックごとのループ
for epoch in range(num_epochs):
	#ネットワークを学習モードにする
	netG_A2B.train(),netG_B2A.train()
	netD_GA.train(),netD_GB.train()
	netD_LA.train(),netD_LB.train()
	#学習過程を追うための変数
	G_losses_per_epoch = []
	D_losses_per_epoch = []
	#データセットA,Bからbatch_size枚ずつ取り出し学習
	for i,(real_A,real_B) in enumerate(zip(dataloader_A,dataloader_B),0):
		#-------------------------
		#学習データの取得
		#-------------------------
		#実際に取得できた学習データのバッチサイズを取得
		minibatch_size_A = real_A.shape[0]
		minibatch_size_B = real_B.shape[0]
		#もしバッチサイズが違っていれば飛ばす
		if(minibatch_size_A != minibatch_size_B): continue
		#実際に取得できた学習データのバッチサイズ
		minibatch_size = minibatch_size_A
		#もしバッチサイズが違っていれば飛ばす
		if(batch_size != minibatch_size): continue
		#GPUが使えるならGPUに転送
		real_A = real_A.to(device)
		real_B = real_B.to(device)
		#real_A : torch.Size([minibatch_size,3,256,256])
		#real_B : torch.Size([minibatch_size,3,256,256])

		#総イテレーション回数の半分以上に達しているならば、学習率を徐々に下げていく
		if(iteration > (total_iteration // 2)):
			optimizerG.param_groups[0]['lr'] -= (learning_rate / (total_iteration//2))
			optimizerD.param_groups[0]['lr'] -= (learning_rate / (total_iteration//2))

		#-------------------------
 		#discriminatorの学習
		#-------------------------
		#前のイテレーションでたまった傾きをリセット
		optimizerD.zero_grad()

		#本物画像から偽物画像を生成
		fake_A2B, _, _ = netG_A2B(real_A)
		fake_B2A, _, _ = netG_B2A(real_B)

		#本物画像に対しそれぞれ判定
		real_GA_logit, real_GA_cam_logit, _ = netD_GA(real_A)
		real_LA_logit, real_LA_cam_logit, _ = netD_LA(real_A)
		real_GB_logit, real_GB_cam_logit, _ = netD_GB(real_B)
		real_LB_logit, real_LB_cam_logit, _ = netD_LB(real_B)

		#偽物画像に対しそれぞれ判定
		fake_GA_logit, fake_GA_cam_logit, _ = netD_GA(fake_B2A)
		fake_LA_logit, fake_LA_cam_logit, _ = netD_LA(fake_B2A)
		fake_GB_logit, fake_GB_cam_logit, _ = netD_GB(fake_A2B)
		fake_LB_logit, fake_LB_cam_logit, _ = netD_LB(fake_A2B)

		#損失の計算
		D_ad_loss_GA = MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(device)) + MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(device))
		D_ad_cam_loss_GA = MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(device)) + MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(device))
		D_ad_loss_LA = MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(device)) + MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(device))
		D_ad_cam_loss_LA = MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(device)) + MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(device))
		D_ad_loss_GB = MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(device)) + MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(device))
		D_ad_cam_loss_GB = MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(device)) + MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(device))
		D_ad_loss_LB = MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(device)) + MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(device))
		D_ad_cam_loss_LB = MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(device)) + MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(device))

		D_loss_A = 1*(D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
		D_loss_B = 1*(D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

		Discriminator_loss = D_loss_A + D_loss_B
		#傾きを計算して
		Discriminator_loss.backward()
		#discriminatorのパラメーターを更新
		optimizerD.step()

		#後でグラフに出力するために記録
		D_losses_per_epoch.append(Discriminator_loss.item())

		#-------------------------
 		#Generatorの学習
		#-------------------------
		#前のイテレーションでたまった傾きをリセット
		optimizerG.zero_grad()

		#本物画像から偽物画像を生成
		fake_A2B, fake_A2B_cam_logit, _ = netG_A2B(real_A)
		fake_B2A, fake_B2A_cam_logit, _ = netG_B2A(real_B)

		#偽物画像から本物に戻ってくるのを目指す
		fake_A2B2A, _, _ = netG_B2A(fake_A2B)
		fake_B2A2B, _, _ = netG_A2B(fake_B2A)

		#変換先と同じドメインの本物画像から偽物画像を生成
		fake_A2A, fake_A2A_cam_logit, _ = netG_B2A(real_A)
		fake_B2B, fake_B2B_cam_logit, _ = netG_A2B(real_B)

		#生成された偽物画像についてそれぞれ判定
		fake_GA_logit, fake_GA_cam_logit, _ = netD_GA(fake_B2A)
		fake_LA_logit, fake_LA_cam_logit, _ = netD_LA(fake_B2A)
		fake_GB_logit, fake_GB_cam_logit, _ = netD_GB(fake_A2B)
		fake_LB_logit, fake_LB_cam_logit, _ = netD_LB(fake_A2B)

		#損失の計算
		G_ad_loss_GA = MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(device))
		G_ad_cam_loss_GA = MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(device))
		G_ad_loss_LA = MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(device))
		G_ad_cam_loss_LA = MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(device))
		G_ad_loss_GB = MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(device))
		G_ad_cam_loss_GB = MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(device))
		G_ad_loss_LB = MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(device))
		G_ad_cam_loss_LB = MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(device))

		G_recon_loss_A = L1_loss(fake_A2B2A, real_A)
		G_recon_loss_B = L1_loss(fake_B2A2B, real_B)

		G_identity_loss_A = L1_loss(fake_A2A, real_A)
		G_identity_loss_B = L1_loss(fake_B2B, real_B)

		G_cam_loss_A = BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(device)) + BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(device))
		G_cam_loss_B = BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(device)) + BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(device))

		G_loss_A =  1*(G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + 10*G_recon_loss_A + 10*G_identity_loss_A + 1000*G_cam_loss_A
		G_loss_B = 1*(G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + 10*G_recon_loss_B + 10*G_identity_loss_B + 1000*G_cam_loss_B

		Generator_loss = G_loss_A + G_loss_B
		#傾きを計算して
		Generator_loss.backward()
		#generatorのパラメーターを更新
		optimizerG.step()

		#後でグラフに出力するために記録
		G_losses_per_epoch.append(Generator_loss.item())

		#学習状況をシェルに出力
		if iteration % 50 == 0:
			print('[%d/%d][iteration:%d]\tLoss_D: %.4f\tLoss_G: %.4f'
					% (epoch,num_epochs,iteration,
						Discriminator_loss.item(),Generator_loss.item()))

		iteration += 1
		#テスト用break
		#break
	
	#後で出力するためにepochごとにlossの平均を取り記録
	G_losses.append(torch.mean(torch.tensor(G_losses_per_epoch,dtype=torch.float64)).item())
	D_losses.append(torch.mean(torch.tensor(D_losses_per_epoch,dtype=torch.float64)).item())
	#Generatorの学習状況を画像として記録
	if (epoch % output_progress_interval == 0 or (epoch+1)==num_epochs):
		#ネットワークを推論モードにする
		netG_A2B.eval(),netG_B2A.eval()
		netD_GA.eval(),netD_GB.eval()
		netD_LA.eval(),netD_LB.eval()
		#画像出力用ディレクトリがなければ作成
		output_dir = "./output/epoch_{}".format(epoch+1)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		#画像の生成と出力
		#デバイスに配置されている画像をcpuに移す関数
		def move_to_cpu(imgs_on_device):
			imgs_on_cpu = []
			for im in imgs_on_device:
				imgs_on_cpu.append(im.detach().cpu())
			return tuple(imgs_on_cpu)
		
		fake_A2B, _, fake_A2B_heatmap = netG_A2B(sample_real_A)
		fake_B2A, _, fake_B2A_heatmap = netG_B2A(sample_real_B)

		fake_A2B2A, _, fake_A2B2A_heatmap = netG_B2A(fake_A2B)
		fake_B2A2B, _, fake_B2A2B_heatmap = netG_A2B(fake_B2A)

		fake_A2B,fake_A2B_heatmap = move_to_cpu([fake_A2B,fake_A2B_heatmap])
		fake_B2A,fake_B2A_heatmap = move_to_cpu([fake_B2A,fake_B2A_heatmap])
		fake_A2B2A,fake_A2B2A_heatmap = move_to_cpu([fake_A2B2A,fake_A2B2A_heatmap])
		fake_B2A2B,fake_B2A2B_heatmap = move_to_cpu([fake_B2A2B,fake_B2A2B_heatmap])

		fake_A2A, _, fake_A2A_heatmap = netG_B2A(sample_real_A)
		fake_B2B, _, fake_B2B_heatmap = netG_A2B(sample_real_B)

		fake_A2A,fake_A2A_heatmap = move_to_cpu([fake_A2A,fake_A2A_heatmap])
		fake_B2B,fake_B2B_heatmap = move_to_cpu([fake_B2B,fake_B2B_heatmap])
		sr_A,sr_B = move_to_cpu([sample_real_A,sample_real_B])
		#A->B->Aの画像の出力
		output_how_much_progress("./output/epoch_{}/conversion_A2B2A.png".format(epoch+1),[sr_A,fake_A2B,fake_A2B2A])
		#B->A->Bの画像の出力
		output_how_much_progress("./output/epoch_{}/conversion_B2A2B.png".format(epoch+1),[sr_B,fake_B2A,fake_B2A2B])
		#ヒートマップ(A)の出力
		fake_A2A_heatmap = F.interpolate(fake_A2A_heatmap,size=(256,256))
		fake_A2B_heatmap = F.interpolate(fake_A2B_heatmap,size=(256,256))
		fake_A2B2A_heatmap = F.interpolate(fake_A2B2A_heatmap,size=(256,256))
		output_how_much_progress("./output/epoch_{}/heatmap_A.png".format(epoch+1).format(epoch+1),[fake_A2A_heatmap,fake_A2B_heatmap,fake_A2B2A_heatmap],normalize=False)
		#ヒートマップ(B)の出力
		fake_B2B_heatmap = F.interpolate(fake_B2B_heatmap,size=(256,256))
		fake_B2A_heatmap = F.interpolate(fake_B2A_heatmap,size=(256,256))
		fake_B2A2B_heatmap = F.interpolate(fake_B2A2B_heatmap,size=(256,256))
		output_how_much_progress("./output/epoch_{}/heatmap_B.png".format(epoch+1).format(epoch+1),[fake_B2B_heatmap,fake_B2A_heatmap,fake_B2A2B_heatmap],normalize=False)
	#テスト用break
	#break

#学習にかかった時間を出力
#学習終了時の時間を記録
t_epoch_finish = time.time()
total_time = t_epoch_finish - t_epoch_start
with open('./output/time.txt', mode='w') as f:
	f.write("total_time: {:.4f} sec.\n".format(total_time))
	f.write("dataset_A size: {}\n".format(len(path_list_A)))
	f.write("dataset_B size: {}\n".format(len(path_list_B)))
	f.write("num_epochs: {}\n".format(num_epochs))
	f.write("batch_size: {}\n".format(batch_size))

#学習済みGeneratorのモデル（CPU向け）を出力
#モデル出力用ディレクトリがなければ作成
output_dir = "./trained_model"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
torch.save(netG_A2B.to('cpu').state_dict(),'./trained_model/generator_A2B_trained_model_cpu.pth')
torch.save(netG_B2A.to('cpu').state_dict(),'./trained_model/generator_B2A_trained_model_cpu.pth')

#lossのグラフを出力
plt.clf()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./output/loss.png')

