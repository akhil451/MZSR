import test
from utils import *
from config import *
import glob
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

conf=tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction=0.95

# def main():
#     if args.model==0:
#         print('Direct Downscaling, Scaling factor x2 Model')
#         model_path = 'Model/Directx2'
#     elif args.model ==1:
#         print('Direct Downscaling, Multi-scale Model')
#         model_path = 'Model/Multi-scale'
#     elif args.model ==2:
#         print('Bicubic Downscaling, Scaling factor x2 Model')
#         model_path = 'Model/Bicubicx2'
#     elif args.model ==3:
#         print('Direct Downscaling, Scaling factor x4 Model')
#         model_path = 'Model/Directx4'

#     img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
#     gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))

#     scale=2.0

#     try:
#         kernel=scipy.io.loadmat(args.kernelpath)['kernel']
#     except:
#         kernel='cubic'

#     Tester=test.Test(model_path, args.savepath, kernel, scale, conf, args.model, args.num_of_adaptation)
#     P=[]
#     for i in range(len(img_path)):
#         img=imread(img_path[i])
#         gt=imread(gt_path[i])

#         _, pp =Tester(img, gt, img_path[i])

#         P.append(pp)

#     avg_PSNR=np.mean(P, 0)

#     print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))


# if __name__=='__main__':
#     main()



import test
from utils import *
from config import *
import glob
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

conf=tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction=0.95

def main():
    if args.model==0:
        print('Direct Downscaling, Scaling factor x2 Model')
        model_path = 'Model/Directx2'
    elif args.model ==1:
        print('Direct Downscaling, Multi-scale Model')
        model_path = 'Model/Multi-scale'
    elif args.model ==2:
        print('Bicubic Downscaling, Scaling factor x2 Model')
        model_path = 'Model/Bicubicx2'
    elif args.model ==3:
        print('Direct Downscaling, Scaling factor x4 Model')
        model_path = 'Model/Directx4'

    img_path=sorted(glob.glob(os.path.join(args.inputpath,"*")))
    gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))

    scale=4.0
    try:
        kernel=scipy.io.loadmat(args.kernelpath)['kernel']
    except:
        kernel='cubic'


    # train_and_save(img_path,gt_path,model_save,kernel,model_save_path)

    Tester=test.Test(model_path, args.savepath, kernel, scale, conf, args.model, args.num_of_adaptation)
    # P=[]
    for i in range(len(img_path)):
        print(img_path[i])
        img=imread(img_path[i])
        # gt=imread(gt_path[i])

        # _, pp =Tester(img, gt, img_path[i])
        # Tester.inference(img,img_path[i])
        output_image= Tester.forward_pass(img,(img.shape[0]*scale,img.shape[1]*scale,3))
        imageio.imsave('%s/%s.png' % (args.savepath, os.path.basename(self.img_name)[:-4]),
                           post_processed_output)
        # P.append(pp)
    # print(P)
    # print("len(P)",len(P))
    # avg_PSNR=np.mean(P, 0)

    # print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))


if __name__=='__main__':
    main()