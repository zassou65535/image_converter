#encoding:utf-8

from module.importer import *
from module.dataloader import *
from module.generator import *

#学習済みモデルの読み込み
generator = Generator()
generator.load_state_dict(torch.load('./trained_model/generator_A2B_trained_model_cpu.pth'))
#推論モードに切り替え
generator.eval()
#変換対象となる画像の読み込み
path_list = make_datapath_list('./convertion/target/**/*')
train_dataset = GAN_Img_Dataset(file_list=path_list,transform=ImageTransform(256))
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=len(path_list),shuffle=False)
target = next(iter(dataloader))
#generatorへ入力、出力画像を得る
converted,_,_ = generator.forward(target)
#画像出力用にディレクトリを作成
os.makedirs("./convertion/converted",exist_ok=True)
#画像を出力
for i,output_img in enumerate(converted):
	origin_filename = os.path.basename(path_list[i])
	origin_filename_without_ex = os.path.splitext(origin_filename)[0]
	filename = "./convertion/converted/{}_converted{}.png".format(origin_filename_without_ex,i)
	#そこへ保存
	vutils.save_image(output_img,filename,normalize=True)
	print(origin_filename + " : converted")




