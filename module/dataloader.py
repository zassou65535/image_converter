#encoding:utf-8

from .importer import *

def make_datapath_list(target_path):
	#読み込むデータセットのパス
	#例えば
	#target_path = "./dataset/**/*"などとします
	#画像のファイル形式はpng
	path_list = []#画像ファイルパスのリストを作り、戻り値とする
	for path in glob.glob(target_path,recursive=True):
		if os.path.isfile(path):
			path_list.append(path)
			##読み込むパスを全部表示　必要ならコメントアウトを外す
			#print(path)
	#読み込んだ画像の数を表示
	print("images : " + str(len(path_list)))
	path_list = sorted(path_list)
	return path_list

class ImageTransform():
	#画像の前処理クラス
	def __init__(self,resize_pixel):
		self.data_transform = transforms.Compose([
				transforms.Resize((resize_pixel,resize_pixel)),
				transforms.ToTensor(),
				transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
			])
	def __call__(self,img):
		return self.data_transform(img)

class ImageModification():
	#画像の前処理クラス　画像の平行移動、拡大縮小なども行う
	def __init__(self,resize_pixel,x_move=[-0.1,0.1],y_move=[-0.1,0.2],min_scale=0.75):
		self.resize_pixel = resize_pixel
		self.x_move = x_move
		self.y_move = y_move
		self.min_scale = min_scale
		self.data_resize = transforms.Resize((resize_pixel*2,resize_pixel*2))
		self.data_arrange = transforms.Compose([
			transforms.Resize((resize_pixel,resize_pixel)),
			transforms.ToTensor(),
			transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
		])
	def __call__(self,img):
		img = self.data_resize(img)
		#中心(y,x)を何pixelずらすかを指定
		move_pixel_x = np.random.uniform(self.resize_pixel*self.x_move[0],self.resize_pixel*self.x_move[1])
		move_pixel_y = np.random.uniform(self.resize_pixel*self.y_move[0],self.resize_pixel*self.y_move[1])
		move_pixel = [move_pixel_x,move_pixel_y]
		#ずらす
		img = transforms.functional.affine(img,angle=0,translate=(move_pixel),scale=1,shear=0)
		#切り取る
		max_crop_size = 2*(self.resize_pixel - np.max(np.abs(move_pixel)))
		min_crop_size = 2*(self.resize_pixel*self.min_scale)
		crop_size = np.random.randint(min_crop_size,max_crop_size)
		img = transforms.functional.center_crop(img,crop_size)
		#Tensorに変換して出力
		img = self.data_arrange(img)
		return img

class GAN_Img_Dataset(data.Dataset):
	#画像のデータセットクラス
	def __init__(self,file_list,transform):
		self.file_list = file_list
		self.transform = transform
	#画像の枚数を返す
	def __len__(self):
		return len(self.file_list)
	#前処理済み画像の、Tensor形式のデータを取得
	def __getitem__(self,index):
		img_path = self.file_list[index]
		img = Image.open(img_path)#[RGB][高さ][幅]
		img = img.convert('RGB')#pngをjpg形式に変換
		img_transformed = self.transform(img)
		return img_transformed

#動作確認
# path_list = make_datapath_list("../dataset/group_B/**/*")

# transform = ImageModification(resize_pixel=256,x_move=[-0.1,0.1],y_move=[-0.1,0.25],min_scale=0.7)
# dataset = GAN_Img_Dataset(file_list=path_list,transform=transform)

# batch_size = 8
# dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)

# imgs = next(iter(dataloader))
# print(imgs.size())

# for i,img_transformed in enumerate(imgs):
# 	img_transformed = img_transformed.detach()
# 	vutils.save_image(img_transformed,"../output/test_img_{}.png".format(i),normalize=True)



